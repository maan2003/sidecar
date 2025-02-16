use std::fmt::Write;
use std::io::{stdout, Write as IoWrite};
use std::{collections::HashMap, sync::Arc};

use anyhow::{bail, Context};
use llm_client::clients::anthropic::AnthropicClient;
use llm_client::clients::types::{LLMClientToolReturn, LLMClientToolUse};
use llm_client::{
    broker::LLMBroker,
    clients::types::{LLMClientCompletionRequest, LLMClientMessage, LLMClientRole, LLMType},
    provider::{LLMProvider, LLMProviderAPIKeys},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::unbounded_channel;
use uuid::Uuid;

use crate::agentic::symbol::{identifier::LLMProperties, tool_box::ToolBox};
use crate::agentic::tool::code_edit::code_editor::{CodeEditorParameters, EditorCommand};
use crate::agentic::tool::session::attempt_completion::AttemptCompletionClientRequest;
use crate::mcts::editor::anthropic_computer::AnthropicCodeEditor;

pub struct Reasoner {
    toolbox: Arc<ToolBox>,
}

impl Reasoner {
    pub fn new(toolbox: Arc<ToolBox>) -> Self {
        Self { toolbox }
    }
}

pub enum Exchange {
    HumanMessage(String),
    Response(String),
}

pub struct HumanMessage {
    pub user_request: String,
    pub context: Vec<RContext>,
}

#[derive(Debug, Clone)]
pub struct Excerpt {
    path: String,
    visible_text: String,
}

#[derive(Debug, Clone)]
pub enum RContext {
    File {
        path: String,
        text: String,
    },
    MultiBuf {
        title: String,
        excerpts: Vec<Excerpt>,
    },
    Knowledge {
        title: String,
        text: String,
    },
    RecentChanges {
        diff: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LLMClientConfig {
    pub models: HashMap<LLMType, LLMProvider>,
    pub providers: Vec<LLMProviderAPIKeys>,
}

impl LLMClientConfig {
    pub fn config_for_llm(&self, llm: LLMType) -> anyhow::Result<LLMProperties> {
        let provider = self
            .models
            .get(&llm)
            .with_context(|| format!("model {llm:?} not configure"))?;
        let api_key = self
            .providers
            .iter()
            .find(|a| &a.provider_type() == provider)
            .context("unconfigured provider")?;
        Ok(LLMProperties::new(llm, provider.clone(), api_key.clone()))
    }
}
// motivation: a clean design free of all legacy code
// question: should we even impl saving/restoring from storage?
pub struct RSession {
    pub id: Uuid,
    pub repo_root: String,
    pub exchanges: Vec<Exchange>,
}

impl RContext {
    fn to_message(&self, s: &mut String) {
        match self {
            RContext::File { path, text } => {
                writeln!(
                    s,
                    "<file path=\"{path}\">\n{text}\n</file>",
                    // TODO: make sure that trim doesn't bite us
                    text = text.trim()
                )
                .unwrap();
            }
            RContext::MultiBuf { title, excerpts } => {
                writeln!(s, "<excerpts title=\"{title}\">").unwrap();
                for ex in excerpts {
                    writeln!(
                        s,
                        "<section file=\"{path}\">\n{text}\n</section>",
                        path = ex.path,
                        text = ex.visible_text.trim()
                    )
                    .unwrap();
                }
                *s += "</excerpts>";
            }
            RContext::Knowledge { title, text } => {
                writeln!(
                    s,
                    "<knowledge title=\"{title}\">\n{text}\n</knowledge>",
                    text = text.trim()
                )
                .unwrap();
            }
            RContext::RecentChanges { diff } => {
                writeln!(
                    s,
                    "<recent_changes>\n{diff}\n</recent_changes>",
                    diff = diff.trim()
                )
                .unwrap();
            }
        }
    }
}
impl RSession {
    pub async fn create_knowledge(
        &mut self,
        request: HumanMessage,
        models_config: &LLMClientConfig,
        llm: &LLMBroker,
    ) -> anyhow::Result<String> {
        let model_props = models_config
            .config_for_llm(LLMType::O3MiniHigh)
            .with_context(|| "O3MiniHigh model not configured")?;

        // Build messages using human message context
        let system_msg = LLMClientMessage::system(
            "You are a technical writer creating detailed documentation for a software project. Generate comprehensive knowledge about the given topic, including examples where applicable.".to_string()
        );

        let mut user_content = String::new();
        user_content += "<context>\n";
        for ctx in &request.context {
            ctx.to_message(&mut user_content);
        }
        user_content += &format!(
            "\n<user_request>\n{}\n</user_request>",
            request.user_request
        );
        let user_msg = LLMClientMessage::user(user_content);

        let request = LLMClientCompletionRequest::new(
            model_props.llm().clone(),
            vec![system_msg, user_msg],
            0.6, // temperature
            None,
        );

        let (sender, _rx) = unbounded_channel();
        let response = llm
            .stream_completion(
                model_props.api_key().clone(),
                request,
                model_props.provider().clone(),
                Default::default(),
                sender,
            )
            .await?;

        Ok(response.answer_up_until_now().to_string())
    }

    fn developer_message() -> LLMClientMessage {
        let text =
        r#"You are a senior software engineer, expert planner and system architect working alongside a software engineer.
- <context> contains the context for the request.
- Context can be a <file path="foo"> with file's entire context.
- Context can be a <excerpts title="References to `Foo::bar()`"> which contains sections of files that reference of an item.
- <previous_messages> (if any) contains the previous user messages and assistant response.
- <recent_changes> (if any) contains recent changes **already applied** to code.
- <user_request> contains the user's request.
- Given a request and context, you will generate a step by step plan to accomplish it. Use prior art seen in context where applicable.
- Your job is to be precise and effective, so avoid extraneous steps even if they offer convenience.
- Use existing patterns in the code unless explicitly requested otherwise.
- Feel free to refactor existing code."#.to_string();
        LLMClientMessage::system(text)
    }

    // might need more stuff
    // run the architect model to reason and generate full instructions for the editor model.
    // run the editor model with full instructions
    // wait for errors from editor
    // run the error fixer model and edit again
    //
    // this is very similar to aider but with additional error fixing step.
    //
    // todo: add deepseek to collect more context.
    // maybe agents to collect context
    // maybe humans are just better? add shortcut to add this file to context and I can goto definition on stuff and add it.
    pub async fn architect_editting(
        &mut self,
        request: HumanMessage,
        models_config: &LLMClientConfig,
        tool_box: &ToolBox,
        llm: &LLMBroker,
    ) -> anyhow::Result<()> {
        let model = models_config.config_for_llm(LLMType::O3MiniHigh)?;
        let user_message = self.build_user_message(&request, true);
        let (sender, mut rx) = unbounded_channel();

        // Start streaming first
        let stream_future = llm.stream_completion(
            model.api_key().clone(),
            LLMClientCompletionRequest::new(
                model.llm().clone(),
                vec![Self::developer_message(), user_message],
                0.6, // ignored by o1
                None,
            )
            .set_max_tokens(20480),
            model.provider().clone(),
            Default::default(),
            sender,
        );

        // Process tokens concurrently
        let processing_task = tokio::spawn(async move {
            while let Some(token) = rx.recv().await {
                print!("{}", token.delta().unwrap_or_default());
                stdout().flush().ok();
            }
        });

        // Wait for both tasks to complete
        let (stream_result, processing_result) = tokio::join!(stream_future, processing_task);
        let (stream_result, _processing_result) = (stream_result?, processing_result?);

        // Append conversation history
        self.exchanges
            .push(Exchange::HumanMessage(request.user_request.clone()));
        self.exchanges.push(Exchange::Response(
            stream_result.answer_up_until_now().to_owned(),
        ));

        // Pass the accumulated response to implementer (without previous exchanges)
        self.implementer(
            HumanMessage {
                user_request: stream_result.answer_up_until_now().to_owned(),
                context: request.context,
            },
            models_config,
            llm,
            false,
        )
        .await?;
        Ok(())
    }

    fn implementer_system() -> LLMClientMessage {
        LLMClientMessage::system(r#"You are an expert software engineer."#.into()).insert_tools(
            vec![
                CodeEditorParameters::to_json(),
                AttemptCompletionClientRequest::to_json(),
            ],
        )
    }

    fn build_user_message(
        &self,
        request: &HumanMessage,
        include_previous_exchanges: bool,
    ) -> LLMClientMessage {
        let mut user_message = String::with_capacity(50_000);

        // If previous exchanges should be included, add them BEFORE the context block
        if include_previous_exchanges && !self.exchanges.is_empty() {
            user_message += "<previous_messages>\n";
            for exchange in &self.exchanges {
                match exchange {
                    Exchange::HumanMessage(msg) => {
                        writeln!(user_message, "<user>{}</user>", msg).unwrap();
                    }
                    Exchange::Response(msg) => {
                        writeln!(user_message, "<assistant>{}</assistant>", msg).unwrap();
                    }
                }
            }
            user_message += "</previous_messages>\n";
        }

        // Add context information from files, knowledge files, git diff, etc.
        user_message += "<context>\n";
        for c in &request.context {
            c.to_message(&mut user_message);
            user_message.push('\n');
        }
        user_message += "</context>\n";

        // Add the actual user request text
        user_message += "<user_request>\n";
        user_message += &request.user_request;
        user_message += "\n</user_request>";
        LLMClientMessage::new(LLMClientRole::User, user_message, vec![])
    }

    // goal: fast editting of file using caching and parallel tool calls
    // we will prefill the messages with editor::view for files that are in context.
    pub async fn implementer(
        &mut self,
        request: HumanMessage,
        models_config: &LLMClientConfig,
        _llm: &LLMBroker,
        include_previous_exchanges: bool,
    ) -> anyhow::Result<()> {
        let model = models_config.config_for_llm(LLMType::ClaudeSonnet)?;
        // Construct user message with context and request
        let user_message = self.build_user_message(&request, include_previous_exchanges);

        let mut messages = vec![Self::implementer_system(), user_message];
        let client = AnthropicClient::new();
        let mut body = String::new();
        let mut tools = vec![];
        'agent: loop {
            // Add cache point for last message
            messages.last_mut().unwrap().set_cache_point(true);

            // remove old cache points
            let mut count = 0;
            for me in messages.iter_mut().rev() {
                if me.is_cache_point() {
                    count += 1;
                }
                // remove cache point from old messages
                if count > 4 {
                    me.set_cache_point(false);
                }
            }
            let (sender, _rx) = unbounded_channel();
            (body, tools) = client
                .stream_completion_with_tool(
                    model.api_key().clone(),
                    LLMClientCompletionRequest::new(
                        model.llm().clone(),
                        messages.clone(),
                        0.0,
                        None,
                    ),
                    Default::default(),
                    sender,
                )
                .await?;
            println!("{body}");
            if tools.is_empty() {
                break;
            }
            // add tool responses
            let mut tool_responses = vec![];
            for (tool, (id, input)) in &tools {
                match &**tool {
                    "str_replace_editor" => {
                        let observation = AnthropicCodeEditor::new(body.clone())
                            .run_command(serde_json::from_str(input)?)
                            .await?;
                        tool_responses.push(LLMClientToolReturn::new(
                            id.to_owned(),
                            tool.to_owned(),
                            observation.message().into(),
                        ));
                    }
                    "attempt_completion" => {
                        println!("done");
                        break 'agent;
                    }
                    _ => bail!("unknown tool: {tool}"),
                }
            }
            messages.push(
                LLMClientMessage::assistant(body).insert_tool_use_values(
                    tools
                        .into_iter()
                        .map(|(name, (id, value))| {
                            LLMClientToolUse::new(name, id, serde_json::from_str(&value).unwrap())
                        })
                        .collect(),
                ),
            );
            messages.push(
                LLMClientMessage::user(String::new()).insert_tool_return_values(tool_responses),
            );
        }
        Ok(())
    }
}
