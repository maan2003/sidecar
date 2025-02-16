use anyhow::{Context as AnyhowContext, Result};
use clap::{Parser, Subcommand};
use reedline::{
    default_emacs_keybindings, ColumnarMenu, DefaultPrompt, DefaultPromptSegment, EditCommand, Emacs, FileBackedHistory,
    KeyCode, KeyModifiers, MenuBuilder as _, Reedline, ReedlineEvent,
};
use std::{collections::HashMap, env, fs, path::PathBuf, sync::Arc};
use tokio::signal;
use xshell::{cmd, Shell};
use rand::{thread_rng, Rng};

// LLM-related imports
use llm_client::{
    broker::LLMBroker,
    clients::types::LLMType,
    provider::{LLMProvider, LLMProviderAPIKeys},
};

// Sidecar imports
use sidecar::{
    agentic::{
        symbol::tool_box::ToolBox,
        tool::broker::{ToolBroker, ToolBrokerConfiguration},
        tool::code_edit::models::broker::CodeEditBroker,
    },
    chunking::{editor_parsing::EditorParsing, languages::TSLanguageParsing},
    inline_completion::symbols_tracker::SymbolTrackerInline,
    webserver::reasoner::{HumanMessage, RContext, RSession},
};

struct JJ {
    original_dir: PathBuf,
    agent_path: PathBuf,
    agent_id: String,
    workspace_root: String,
    sh: Shell,
}

impl JJ {
    fn new() -> Result<Self> {
        let original_dir = env::current_dir()?;
        let sh = Shell::new()?;

        let workspace_root_str = cmd!(sh, "jj workspace root").read()?.trim().to_string();
        let workspace_root = PathBuf::from(&workspace_root_str);

        let agent_root = workspace_root.join(".jj").join("agent");
        fs::create_dir_all(&agent_root)?;

        let (agent_id, agent_path) = {
            let mut rng = thread_rng();
            loop {
                let candidate: String = (0..4)
                    .map(|_| rng.gen_range(b'a'..=b'z') as char)
                    .collect();
                let candidate_path = agent_root.join(&candidate);
                if !candidate_path.exists() {
                    break (candidate, candidate_path);
                }
            }
        };
        cmd!(sh, "jj workspace add {agent_path}").run()?;
        env::set_current_dir(&agent_path)?;
        sh.change_dir(&agent_path);

        Ok(JJ {
            original_dir,
            agent_path,
            agent_id,
            workspace_root: workspace_root_str,
            sh,
        })
    }

    fn record(&self) -> Result<()> {
        cmd!(self.sh, "jj st").run()?;
        Ok(())
    }

    fn get_diff(&self) -> Result<Option<String>> {
        let diff_text = cmd!(self.sh, "jj diff --git").read()?;
        if diff_text.trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(diff_text))
        }
    }

    fn run_diff(&self) -> Result<()> {
        cmd!(self.sh, "jj diff").run()?;
        Ok(())
    }

    fn cleanup(&self) -> Result<()> {
        env::set_current_dir(&self.original_dir)?;
        self.sh.change_dir(&self.original_dir);

        let agent_id = &self.agent_id;
        cmd!(self.sh, "jj workspace forget {agent_id}").run()?;

        fs::remove_dir_all(&self.agent_path)?;

        Ok(())
    }
}

impl Drop for JJ {
    fn drop(&mut self) {
        if let Err(err) = self.cleanup() {
            eprintln!("Error during JJ cleanup: {}", err);
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "", disable_help_subcommand = true, disable_help_flag = true)]
struct Command {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Exit the REPL
    Exit,
    /// Show current context
    Context,
    /// Add file to context
    Add {
        /// Path to the file to add
        #[arg(required = true)]
        file_path: PathBuf,
    },
    /// Remove file from context by index
    Remove {
        /// Index of the file to remove
        #[arg(required = true)]
        index: usize,
    },
    /// Clear all pending files
    ClearContext,
    /// Clear exchange history
    Clear,
    /// Create a new knowledge file using O3-Mini-High model
    CreateKnowledge {
        /// Title of the knowledge file to create (without .md extension)
        #[arg(required = true)]
        title: String,
    },
    /// Load a knowledge file by title (without .md extension)
    LoadKnowledge {
        /// Title of the knowledge file to load
        #[arg(required = true)]
        title: String,
    },
    /// Unload a knowledge file by title
    UnloadKnowledge {
        /// Title of the knowledge file to unload
        #[arg(required = true)]
        title: String,
    },
    /// List loaded knowledge files
    ListKnowledge,
    /// Direct implementer request (bypasses architect)
    Implementer {
        /// The request to send directly to implementer
        #[arg(required = true)]
        request: String,
    },
    /// Include recent changes from git diff
    IncludeRecentChanges,
    /// Run 'jj diff' and output the diff to the terminal
    Diff,
    /// Generate a commit message by summarizing changes using the current git diff.
    CommitMessage,
    /// Run a shell command and add its output as context
    RunCommand {
        /// Run a shell command and add its output as context
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    Help,
}

// Helper function to print command contexts
fn print_command_contexts(pending_commands: &[RContext]) {
    if pending_commands.is_empty() {
        println!("No command outputs in context.");
        return;
    }
    println!("\nPending command outputs:");
    for (idx, ctx) in pending_commands.iter().enumerate() {
        if let RContext::Command { command, .. } = ctx {
            println!("[{}] Command: {}", idx, command);
        }
    }
}

// Helper function to print current context
fn print_context(pending_files: &[String]) {
    if pending_files.is_empty() {
        println!("No pending files.");
        return;
    }
    println!("\nPending files:");
    for (idx, path) in pending_files.iter().enumerate() {
        println!("[{}] {}", idx, path);
    }
}

// Helper function to print help
fn print_help() {
    println!("\nAvailable commands:");
    println!("  help                 - Show this help message");
    println!("  exit                 - Exit the REPL");
    println!("  context              - Show pending files and loaded knowledge files");
    println!("  add <file_path>      - Add file to pending files");
    println!("  remove <index>       - Remove file from pending files by index");
    println!("  clear_context        - Clear all pending files");
    println!("  clear                - Clear exchange history");
    println!("  create_knowledge <title> - Create a new knowledge file using DeepSeekR1 model");
    println!("  load_knowledge <title> - Load a knowledge file by title (without .md extension)");
    println!("  unload_knowledge <title> - Unload a knowledge file by title");
    println!("  list_knowledge       - List loaded knowledge files");
    println!("  implementer <request> - Send request directly to implementer (bypass architect)");
    println!("  include_recent_changes - Include recent changes from git diff");
    println!("  diff                 - Run 'jj diff' and output the diff to terminal");
    println!("  commit_message             - Generate a commit message using the current git diff");
    println!("  run_command <command> - Run a shell command and add its output as context");
    println!("\nAny other input will be processed as a request to the reasoner.");
    println!("When processing a request, all pending files and loaded knowledge files will be used as context.");
    println!("Tip: Press Ctrl+E to prefix current line with 'implementer ' command.");
}

struct Complete;
impl reedline::Completer for Complete {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<reedline::Suggestion> {
        use clap::CommandFactory;
        let mut command = Command::command();
        let start_str = &line[..pos];
        let mut args = shlex::split(start_str).unwrap_or_default();
        let last = start_str.is_empty() || start_str.ends_with(" ");
        args.insert(0, "reson".into());
        if last {
            args.push(String::new());
        }
        let len = args.len();
        let last_len = args.last().map_or(0, String::len);
        clap_complete::engine::complete(
            &mut command,
            args.into_iter().map(Into::into).collect(),
            len - 1,
            Some(&std::env::current_dir().unwrap()),
        )
        .ok()
        .into_iter()
        .flatten()
        .map(|c| reedline::Suggestion {
            value: c.get_value().to_string_lossy().into_owned(),
            description: c.get_help().map(|s| s.ansi().to_string()),
            style: None,
            extra: None,
            span: reedline::Span {
                start: pos - last_len,
                end: pos,
            },
            append_whitespace: false,
        })
        .collect()
    }
}
// Helper function to print loaded knowledge
fn print_loaded_knowledge(loaded_knowledge: &[String]) {
    if loaded_knowledge.is_empty() {
        println!("No knowledge files loaded.");
        return;
    }
    println!("\nLoaded knowledge files:");
    for title in loaded_knowledge {
        println!("  {}", title);
    }
}

// Helper function to get git diff if needed
async fn maybe_get_git_diff(jj: &JJ, include_recent_changes: bool) -> Result<Option<String>> {
    if !include_recent_changes {
        return Ok(None);
    }
    jj.get_diff()
}


// Helper function to build HumanMessage with context
async fn build_human_message(
    jj: &JJ,
    pending_file_paths: &[String],
    loaded_knowledge: &[String],
    pending_command_contexts: &[RContext],
    recent_changes_flag: bool,
    knowledge_dir: &PathBuf,
    request: String,
) -> Result<HumanMessage> {
    let mut context = vec![];

    // Load pending files
    for file_path in pending_file_paths {
        match tokio::fs::read_to_string(file_path).await {
            Ok(text) => {
                context.push(RContext::File {
                    path: file_path.to_string(),
                    text,
                });
            }
            Err(e) => {
                eprintln!("Error reading file {}: {}", file_path, e);
            }
        }
    }

    // Load knowledge files
    for title in loaded_knowledge {
        let file_path = knowledge_dir.join(format!("{}.md", title));
        if file_path.exists() {
            match tokio::fs::read_to_string(&file_path).await {
                Ok(text) => {
                    context.push(RContext::Knowledge {
                        title: title.clone(),
                        text,
                    });
                }
                Err(e) => {
                    eprintln!("Error reading knowledge file {}: {}", title, e);
                }
            }
        }
    }

    // Add command contexts
    for cmd_ctx in pending_command_contexts.iter() {
        context.push(cmd_ctx.clone());
    }

    // Add recent changes if needed
    if let Some(diff) = maybe_get_git_diff(jj, recent_changes_flag).await? {
        context.push(RContext::RecentChanges { diff });
    }

    Ok(HumanMessage {
        user_request: request,
        context,
    })
}

// Helper function to process commands
async fn process_input(
    line: &str,
    session: &mut RSession,
    models_config: &sidecar::webserver::reasoner::LLMClientConfig,
    tool_box: &ToolBox,
    llm: &Arc<llm_client::broker::LLMBroker>,
    pending_file_paths: &mut Vec<String>,
    pending_command_contexts: &mut Vec<RContext>,
    knowledge_dir: &PathBuf,
    loaded_knowledge: &mut Vec<String>,
    recent_changes_flag: &mut bool,
    jj: &JJ,
) -> Result<bool> {
    // Parse the input line as if it were command line arguments
    let mut args = shlex::split(line).unwrap_or_default();
    args.insert(0, "reson".into());
    let command = match Command::try_parse_from(args) {
        Ok(cmd) => cmd,
        Err(_e) => {
            let request = build_human_message(
                jj,
                pending_file_paths,
                &*loaded_knowledge,
                pending_command_contexts,
                *recent_changes_flag,
                knowledge_dir,
                line.to_string(),
            )
            .await?;

            session
                .architect_editting(request, models_config, tool_box, llm)
                .await?;
            jj.record()?;
            println!("Request processed successfully.");
            return Ok(false);
        }
    };

    match command.command {
        Commands::Exit => Ok(true),
        Commands::Context => {
            print_context(pending_file_paths);
            print_loaded_knowledge(loaded_knowledge);
            print_command_contexts(pending_command_contexts);
            Ok(false)
        }
        Commands::Add { file_path } => {
            if file_path.exists() {
                pending_file_paths.push(file_path.to_string_lossy().to_string());
                println!("Added file to pending context: {}", file_path.display());
            } else {
                println!("File not found: {}", file_path.display());
            }
            Ok(false)
        }
        Commands::Remove { index } => {
            if index < pending_file_paths.len() {
                pending_file_paths.remove(index);
                println!("Removed file at index {}", index);
            } else {
                println!("Index {} is out of range", index);
            }
            Ok(false)
        }
        Commands::ClearContext => {
            pending_file_paths.clear();
            pending_command_contexts.clear();
            println!("Cleared all pending files and command outputs");
            Ok(false)
        }
        Commands::Clear => {
            session.exchanges.clear();
            println!("Cleared exchange history.");
            Ok(false)
        }
        Commands::CreateKnowledge { title } => {
            let file_path = knowledge_dir.join(format!("{}.md", title));

            // Build human message with context
            let human_message = build_human_message(
                jj,
                pending_file_paths,
                &*loaded_knowledge,
                pending_command_contexts,
                *recent_changes_flag,
                knowledge_dir,
                format!("Generate comprehensive documentation about '{}'", title),
            )
            .await?;

            // Generate content using RSession
            let content = session
                .create_knowledge(human_message, models_config, llm)
                .await?;

            // Save to file
            tokio::fs::write(&file_path, content.trim()).await?;

            // Add to loaded_knowledge if not already present
            if !loaded_knowledge.contains(&title) {
                loaded_knowledge.push(title.clone());
            }

            println!("Created and loaded knowledge file: {}", title);
            Ok(false)
        }
        Commands::LoadKnowledge { title } => {
            let file_path = knowledge_dir.join(format!("{}.md", title));
            if file_path.exists() {
                if !loaded_knowledge.contains(&title) {
                    loaded_knowledge.push(title.clone());
                    println!("Loaded knowledge file: {}", title);
                } else {
                    println!("Knowledge file already loaded: {}", title);
                }
            } else {
                println!("Knowledge file not found: {}", title);
            }
            Ok(false)
        }
        Commands::UnloadKnowledge { title } => {
            if let Some(pos) = loaded_knowledge.iter().position(|x| x == &title) {
                loaded_knowledge.remove(pos);
                println!("Unloaded knowledge file: {}", title);
            } else {
                println!("Knowledge file not loaded: {}", title);
            }
            Ok(false)
        }
        Commands::ListKnowledge => {
            print_loaded_knowledge(loaded_knowledge);
            Ok(false)
        }
        Commands::Implementer { request } => {
            let request = build_human_message(
                jj,
                pending_file_paths,
                &*loaded_knowledge,
                pending_command_contexts,
                *recent_changes_flag,
                knowledge_dir,
                request,
            )
            .await?;

            session
                .implementer(request, models_config, llm, true)
                .await?;
            jj.record()?;
            println!("Request processed successfully.");
            Ok(false)
        }
        Commands::Help => {
            print_help();
            Ok(false)
        }
        Commands::IncludeRecentChanges => {
            *recent_changes_flag = !*recent_changes_flag;
            if *recent_changes_flag {
                println!("Recent changes will be included in next request");
            } else {
                println!("Recent changes will not be included in next request");
            }
            Ok(false)
        }
        Commands::Diff => {
            if let Err(e) = jj.run_diff() {
                eprintln!("Error running diff: {}", e);
            }
            Ok(false)
        }
        Commands::CommitMessage => {
            let diff_text = match jj.get_diff() {
                Ok(Some(d)) => d,
                Ok(None) => {
                    println!("No changes found in git diff.");
                    return Ok(false);
                }
                Err(e) => {
                    eprintln!("Error fetching git diff: {}", e);
                    return Ok(false);
                }
            };

            // Create a HumanMessage incorporating the recent changes diff.
            let human_message = HumanMessage {
                user_request: "Generate commit message".to_string(),
                context: vec![RContext::RecentChanges { diff: diff_text }],
            };

            let commit_msg = session
                .generate_commit_message(human_message, models_config, &*llm)
                .await?;
            println!("Generated commit message:\n{}", commit_msg);
            Ok(false)
        }
        Commands::RunCommand { command } => {
            if command.is_empty() {
                println!("No command provided.");
                return Ok(false);
            }
            let mut cmd = tokio::process::Command::new(&command[0]);
            for arg in command.iter().skip(1) {
                cmd.arg(arg);
            }
            // Inherit stderr so that it prints directly to the terminal
            cmd.stderr(std::process::Stdio::inherit());
            let output = cmd.output().await?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            pending_command_contexts.push(RContext::Command {
                command: command.join(" "),
                output: stdout,
            });
            println!("Command output added to context");
            Ok(false)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let jj = JJ::new().expect("Failed to set up agent workspace");

    let language_parsing = Arc::new(TSLanguageParsing::init());
    let llm_broker = Arc::new(LLMBroker::new().await?);
    let editor_parsing = Arc::new(EditorParsing::default());
    let symbol_tracker = Arc::new(SymbolTrackerInline::new(editor_parsing.clone()));

    // Setup LLM configuration
    let models_config = sidecar::webserver::reasoner::LLMClientConfig {
        models: HashMap::from_iter([
            (LLMType::O3MiniHigh, LLMProvider::OpenAI),
            (LLMType::Gpt4O, LLMProvider::OpenAI),
            (LLMType::ClaudeSonnet, LLMProvider::Anthropic),
            (LLMType::DeepSeekR1, LLMProvider::FireworksAI),
        ]),
        providers: vec![
            LLMProviderAPIKeys::OpenAI(llm_client::provider::OpenAIProvider {
                api_key: env::var("OPENAI_API_KEY").context("OPENAI_API_KEY not set")?,
            }),
            LLMProviderAPIKeys::Anthropic(llm_client::provider::AnthropicAPIKey {
                api_key: env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY not set")?,
            }),
            LLMProviderAPIKeys::FireworksAI(llm_client::provider::FireworksAPIKey {
                api_key: env::var("FIREWORKS_API_KEY").context("FIREWORKS_API_KEY not set")?,
            }),
        ],
    };

    let tool_broker = Arc::new(
        ToolBroker::new(
            llm_broker.clone(),
            Arc::new(CodeEditBroker::new()),
            symbol_tracker.clone(),
            language_parsing.clone(),
            ToolBrokerConfiguration::new(None, true),
            models_config.config_for_llm(LLMType::Gpt4O).unwrap(),
        )
        .await,
    );
    let tool_box = Arc::new(ToolBox::new(
        tool_broker.clone(),
        symbol_tracker.clone(),
        editor_parsing.clone(),
    ));

    // Initialize session, pending files vector and knowledge directory
    let mut pending_file_paths = Vec::new();
    let mut pending_command_contexts = Vec::new();
    let dirs = directories::ProjectDirs::from("com", "sidecar", "reasoner")
        .expect("Could not determine project directories");
    let knowledge_dir = dirs.data_dir().join("knowledge");
    if !knowledge_dir.exists() {
        tokio::fs::create_dir_all(&knowledge_dir).await?;
    }
    let mut session = RSession {
        id: uuid::Uuid::new_v4(),
        repo_root: ".".to_string(),
        exchanges: vec![],
    };
    let mut loaded_knowledge = Vec::new();

    let mut include_recent_changes_flag = false;

    // Create history file path and initialize file-backed history
    let history_file = dirs.data_dir().join("reasoner_history.txt");
    let history = FileBackedHistory::with_file(1000, history_file)?;

    // Initialize ReedLine
    let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );
    // Add Ctrl+E binding to insert "implementer " at start of line
    keybindings.add_binding(
        KeyModifiers::CONTROL,
        KeyCode::Char('e'),
        ReedlineEvent::Multiple(vec![
            ReedlineEvent::ClearScreen,
            ReedlineEvent::Edit(vec![EditCommand::InsertString("implementor ".to_string())]),
        ]),
    );
    let edit_mode = Box::new(Emacs::new(keybindings));
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic(jj.workspace_root.to_owned()),
        DefaultPromptSegment::Empty,
    );

    let mut line_editor = Reedline::create()
        .with_completer(Box::new(Complete))
        .with_menu(reedline::ReedlineMenu::EngineCompleter(completion_menu))
        .with_edit_mode(edit_mode)
        .with_history(Box::new(history));

    println!("Welcome to the Reasoner REPL. Type 'help' for available commands.");

    // REPL loop
    loop {
        match line_editor.read_line(&prompt) {
            Ok(reedline::Signal::Success(line)) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let process_result = tokio::select! {
                    res = process_input(
                        line,
                        &mut session,
                        &models_config,
                        &tool_box,
                        &llm_broker,
                        &mut pending_file_paths,
                        &mut pending_command_contexts,
                        &knowledge_dir,
                        &mut loaded_knowledge,
                        &mut include_recent_changes_flag,
                        &jj,
                    ) => Some(res),
                    _ = signal::ctrl_c() => None,
                };

                match process_result {
                    Some(Ok(true)) => break,
                    Some(Ok(false)) => continue,
                    Some(Err(e)) => {
                        eprintln!("Error processing input: {}", e);
                        continue;
                    }
                    // Ctrl+C was pressed while process_input was running
                    None => {
                        println!("Operation canceled.");
                        continue;
                    }
                }
            }
            Ok(reedline::Signal::CtrlC | reedline::Signal::CtrlD) => {
                break;
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
                break;
            }
        }
    }

    Ok(())
}
