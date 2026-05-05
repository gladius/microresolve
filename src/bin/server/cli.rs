//! CLI argument parsing + config file loading.
//!
//! Merge priority (highest wins):
//!   1. CLI flag
//!   2. Environment variable
//!   3. Config file (~/.config/microresolve/config.toml)
//!   4. Built-in default
//!
//! Recognized env vars: `MICRORESOLVE_PORT`, `MICRORESOLVE_HOST`,
//! `MICRORESOLVE_DATA_DIR`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`,
//! `ANTHROPIC_API_KEY` (fallback for `LLM_API_KEY`).

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "microresolve",
    version,
    about = "Pre-LLM decision layer: intent classification, tool selection, request triage.",
    long_about = "MicroResolve — a microsecond classical classifier for the pre-LLM decision layer.\n\n\
        Run without arguments to start the server with defaults (http://localhost:4000).\n\
        Set up persistent config (API keys, etc.) with:  microresolve config\n\n\
        Homepage: https://github.com/gladius/microresolve"
)]
pub struct Cli {
    /// Port to listen on (default: 4000). Overrides MICRORESOLVE_PORT env var and config file.
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,

    /// Host/interface to bind to (default: 0.0.0.0).
    #[arg(long, value_name = "HOST")]
    pub host: Option<String>,

    /// Data directory for persistent state (default: ~/.local/share/microresolve).
    #[arg(long, value_name = "DIR")]
    pub data: Option<PathBuf>,

    /// LLM API key for training/auto-learn features. If omitted, these features are disabled.
    /// Prefer setting this via `microresolve config` or the LLM_API_KEY env var.
    #[arg(long, value_name = "KEY")]
    pub llm_key: Option<String>,

    /// LLM provider: anthropic | gemini | openai.
    #[arg(long, value_name = "PROVIDER")]
    pub llm_provider: Option<String>,

    /// LLM model id.
    #[arg(long, value_name = "MODEL")]
    pub llm_model: Option<String>,

    /// Don't auto-launch a browser at the Studio URL on startup.
    /// Useful on headless servers, in CI, or when you already have a tab open.
    #[arg(long)]
    pub no_browser: bool,

    /// API keys file (default: ~/.config/microresolve/keys.json). Override
    /// for tests or for running multiple Studios with isolated keystores.
    #[arg(long, value_name = "FILE")]
    pub keys_file: Option<PathBuf>,

    /// Config file path (default: ~/.config/microresolve/config.toml). Override
    /// for sandbox tests or for running multiple Studios with isolated
    /// per-instance configs. Also honored by `microresolve config` for writes.
    /// `MICRORESOLVE_CONFIG` env var works as a fallback.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Print the resolved configuration (after merging CLI/env/file) and exit.
    #[arg(long)]
    pub print_config: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Interactively set up a persistent config file at ~/.config/microresolve/config.toml.
    Config,
    /// Install a reference pack from the GitHub release matching this binary's version.
    ///
    /// The pack tarball is fetched from:
    ///   https://github.com/gladius/microresolve/releases/download/v<VERSION>/pack-<NAME>-v<VERSION>.tar.gz
    ///
    /// Available packs: safety-filter, eu-ai-act-prohibited, hipaa-triage, mcp-tools-generic
    Install {
        /// Pack name (e.g. safety-filter, hipaa-triage, eu-ai-act-prohibited, mcp-tools-generic)
        pack: String,
    },
    /// List the 4 reference packs and show install status against the configured data dir.
    ListPacks,
}

/// The on-disk config file. All fields are optional; missing fields fall back to
/// env vars or built-in defaults.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigFile {
    pub port: Option<u16>,
    pub host: Option<String>,
    pub data_dir: Option<PathBuf>,
    pub llm_provider: Option<String>,
    pub llm_model: Option<String>,
    pub llm_api_key: Option<String>,
}

/// Resolved runtime configuration after merging CLI > env > file > defaults.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub port: u16,
    pub host: String,
    pub data_dir: PathBuf,
    pub llm_provider: String,
    pub llm_model: String,
    pub llm_api_key: Option<String>,
    pub no_browser: bool,
    /// `Some` only when `--keys-file` was passed explicitly. `None` means
    /// "use the platform default" (`~/.config/microresolve/keys.json`).
    pub keys_file: Option<PathBuf>,
}

/// Path to the user's config file (created on first `microresolve config` run).
/// Default location only — see `resolve_config_path` for the cascade.
pub fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("sh", "gladius", "microresolve")
        .map(|pd| pd.config_dir().join("config.toml"))
}

/// Resolve the config file path via cascade: `--config` flag > `MICRORESOLVE_CONFIG`
/// env > XDG default. Returns `None` only if neither override is set and the
/// XDG default can't be determined.
pub fn resolve_config_path(cli: &Cli) -> Option<PathBuf> {
    cli.config
        .clone()
        .or_else(|| std::env::var("MICRORESOLVE_CONFIG").ok().map(PathBuf::from))
        .or_else(config_path)
}

/// Default data directory (XDG-ish).
pub fn default_data_dir() -> PathBuf {
    directories::ProjectDirs::from("sh", "gladius", "microresolve")
        .map(|pd| pd.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("./microresolve-data"))
}

/// Load the config file at the resolved path if it exists, return default otherwise.
pub fn load_config_file(cli: &Cli) -> ConfigFile {
    let Some(path) = resolve_config_path(cli) else {
        return ConfigFile::default();
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return ConfigFile::default();
    };
    toml::from_str::<ConfigFile>(&content).unwrap_or_default()
}

/// Merge all sources: CLI flag > env > config file > built-in default.
pub fn resolve(cli: &Cli) -> ResolvedConfig {
    let file = load_config_file(cli);

    let port = cli
        .port
        .or_else(|| {
            std::env::var("MICRORESOLVE_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .or(file.port)
        .unwrap_or(4000);

    let host = cli
        .host
        .clone()
        .or_else(|| std::env::var("MICRORESOLVE_HOST").ok())
        .or(file.host)
        .unwrap_or_else(|| "0.0.0.0".to_string());

    let data_dir = cli
        .data
        .clone()
        .or_else(|| {
            std::env::var("MICRORESOLVE_DATA_DIR")
                .ok()
                .map(PathBuf::from)
        })
        .or(file.data_dir)
        .unwrap_or_else(default_data_dir);

    let llm_provider = cli
        .llm_provider
        .clone()
        .or_else(|| std::env::var("LLM_PROVIDER").ok())
        .or(file.llm_provider)
        .unwrap_or_else(|| "anthropic".to_string());

    let llm_model = cli
        .llm_model
        .clone()
        .or_else(|| std::env::var("LLM_MODEL").ok())
        .or(file.llm_model)
        .unwrap_or_else(|| match llm_provider.as_str() {
            "gemini" => "gemini-2.5-flash".to_string(),
            "openai" => "gpt-4o-mini".to_string(),
            _ => "claude-haiku-4-5-20251001".to_string(),
        });

    let llm_api_key = cli
        .llm_key
        .clone()
        .or_else(|| std::env::var("LLM_API_KEY").ok())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .or(file.llm_api_key);

    ResolvedConfig {
        port,
        host,
        data_dir,
        llm_provider,
        llm_model,
        llm_api_key,
        no_browser: cli.no_browser,
        keys_file: cli.keys_file.clone(),
    }
}

/// Pretty-print the resolved config to stdout (for --print-config).
pub fn print_resolved(cfg: &ResolvedConfig, cli: &Cli) {
    println!("Resolved configuration:");
    println!("  host         = {}", cfg.host);
    println!("  port         = {}", cfg.port);
    println!("  data_dir     = {}", cfg.data_dir.display());
    println!("  llm_provider = {}", cfg.llm_provider);
    println!("  llm_model    = {}", cfg.llm_model);
    println!(
        "  llm_api_key  = {}",
        if cfg.llm_api_key.is_some() {
            "(set, hidden)"
        } else {
            "(not set — training features disabled)"
        }
    );
    if let Some(p) = resolve_config_path(cli) {
        println!("  config_file  = {}", p.display());
    }
}

/// Interactive setup: prompt the user for key fields, write them to the config file.
/// Honors `--config <PATH>` and `MICRORESOLVE_CONFIG`; falls back to the XDG default.
pub fn run_config_subcommand(cli: &Cli) -> std::io::Result<()> {
    use std::io::{BufRead, Write};

    let path = resolve_config_path(cli).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "could not determine config directory",
        )
    })?;

    println!("MicroResolve configuration setup");
    println!("Will write to: {}\n", path.display());

    let existing = load_config_file(cli);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();
    let mut buf = String::new();

    let prompt = |stdout: &mut std::io::StdoutLock,
                  stdin: &mut std::io::StdinLock,
                  buf: &mut String,
                  label: &str,
                  current: Option<&str>|
     -> std::io::Result<Option<String>> {
        match current {
            Some(c) => write!(stdout, "{} [{}]: ", label, c)?,
            None => write!(stdout, "{}: ", label)?,
        }
        stdout.flush()?;
        buf.clear();
        stdin.read_line(buf)?;
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            Ok(current.map(|s| s.to_string()))
        } else {
            Ok(Some(trimmed.to_string()))
        }
    };

    let llm_provider = prompt(
        &mut stdout,
        &mut stdin,
        &mut buf,
        "LLM provider (anthropic/gemini/openai)",
        existing.llm_provider.as_deref().or(Some("anthropic")),
    )?;
    let llm_model = prompt(
        &mut stdout,
        &mut stdin,
        &mut buf,
        "LLM model (leave blank to use provider default)",
        existing.llm_model.as_deref(),
    )?;
    let llm_api_key = prompt(
        &mut stdout,
        &mut stdin,
        &mut buf,
        "LLM API key (leave blank to skip — training features will be disabled)",
        existing
            .llm_api_key
            .as_deref()
            .map(|_| "(existing, keep as-is)"),
    )?;
    // If user pressed enter on the "keep as-is" sentinel, preserve the existing key.
    let llm_api_key = match llm_api_key.as_deref() {
        Some("(existing, keep as-is)") => existing.llm_api_key.clone(),
        other => other.map(|s| s.to_string()),
    };

    let port = prompt(
        &mut stdout,
        &mut stdin,
        &mut buf,
        "Port (blank = 4000)",
        existing.port.map(|p| p.to_string()).as_deref(),
    )?
    .and_then(|s| s.parse().ok());

    let data_dir = prompt(
        &mut stdout,
        &mut stdin,
        &mut buf,
        "Data directory (blank = default)",
        existing
            .data_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .as_deref(),
    )?
    .map(PathBuf::from);

    let new_config = ConfigFile {
        port,
        host: existing.host,
        data_dir,
        llm_provider,
        llm_model,
        llm_api_key,
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_str =
        toml::to_string_pretty(&new_config).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(&path, toml_str)?;

    // Best-effort: make config file user-read-only (the API key is sensitive).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    println!("\nWrote {}", path.display());
    Ok(())
}

/// Detect if we're probably running in a headless environment (SSH, container, CI).
/// We won't auto-open the browser there.
pub fn looks_headless() -> bool {
    if std::env::var_os("SSH_CONNECTION").is_some()
        || std::env::var_os("SSH_CLIENT").is_some()
        || std::env::var_os("SSH_TTY").is_some()
    {
        return true;
    }
    // CI markers
    if std::env::var_os("CI").is_some() || std::env::var_os("GITHUB_ACTIONS").is_some() {
        return true;
    }
    // Linux without DISPLAY/WAYLAND
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
            return true;
        }
    }
    false
}
