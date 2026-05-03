use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;

use serde_json::{json, Value};
use va_ai_api_proxy::{
    AnthropicMessagesTranslator, OpenAiChatTranslator, OpenAiResponsesTranslator, Result,
    UniversalRequest, WireProtocol, WireTranslator,
};

const DEFAULT_PROMPT: &str = "Reply exactly with: va-api-proxy-ok";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    let profile = Profile::load(args.profile_id.as_deref())?;
    let protocol = args.protocol.unwrap_or_else(|| {
        if profile
            .api_types
            .iter()
            .any(|api_type| api_type == "openai-responses")
        {
            WireProtocol::OpenAiResponses
        } else if profile
            .api_types
            .iter()
            .any(|api_type| api_type == "anthropic")
        {
            WireProtocol::AnthropicMessages
        } else {
            WireProtocol::OpenAiChat
        }
    });
    let prompt = args.prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let endpoint = args
        .url
        .unwrap_or_else(|| endpoint_for(protocol, &profile.base_url(protocol)));
    let model = args.model.unwrap_or_else(|| profile.model(protocol));
    let request = request_for(protocol, &model, &prompt);
    let translator = translator_for(protocol);

    let mut universal_request = translator.decode_request(request.clone())?;
    strip_source_raw(&mut universal_request);

    let raw_response = post_json(protocol, &endpoint, &profile.api_key, &request)?;
    let response_events = translator.decode_response(raw_response.clone())?;

    let output = json!({
        "protocol": protocol.as_str(),
        "profile": {
            "id": profile.id,
            "label": profile.label,
        },
        "endpoint": endpoint,
        "wireRequest": request,
        "universalRequest": universal_request,
        "rawResponse": raw_response,
        "universalResponseEvents": response_events,
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    Ok(())
}

struct Args {
    profile_id: Option<String>,
    protocol: Option<WireProtocol>,
    prompt: Option<String>,
    url: Option<String>,
    model: Option<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = env::args().skip(1);
        let mut parsed = Self {
            profile_id: None,
            protocol: None,
            prompt: None,
            url: None,
            model: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--profile" => parsed.profile_id = next_arg(&mut args, "--profile")?,
                "--protocol" => {
                    let value = next_arg(&mut args, "--protocol")?.unwrap();
                    parsed.protocol = Some(parse_protocol(&value)?);
                }
                "--prompt" => parsed.prompt = next_arg(&mut args, "--prompt")?,
                "--url" => parsed.url = next_arg(&mut args, "--url")?,
                "--model" => parsed.model = next_arg(&mut args, "--model")?,
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(va_ai_api_proxy::ApiProxyError::invalid_request(format!(
                        "unknown argument: {other}"
                    )));
                }
            }
        }

        Ok(parsed)
    }
}

struct Profile {
    id: String,
    label: String,
    api_types: Vec<String>,
    api_key: String,
    raw: Value,
}

impl Profile {
    fn load(profile_id: Option<&str>) -> Result<Self> {
        let path = match profile_id {
            Some(profile_id) => profiles_dir().join(format!("{profile_id}.json")),
            None => find_custom_profile()?,
        };
        let body = std::fs::read_to_string(&path).map_err(|error| {
            va_ai_api_proxy::ApiProxyError::invalid_request(format!(
                "read profile {}: {error}",
                path.display()
            ))
        })?;
        let raw: Value = serde_json::from_str(&body).map_err(|error| {
            va_ai_api_proxy::ApiProxyError::invalid_request(format!(
                "parse profile {}: {error}",
                path.display()
            ))
        })?;
        let api_key = raw
            .pointer("/credentials/api_key")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                va_ai_api_proxy::ApiProxyError::invalid_request(format!(
                    "profile {} has no credentials.api_key",
                    path.display()
                ))
            })?
            .to_string();
        let api_types = raw
            .get("api_types")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            id: string_field(&raw, "id").unwrap_or_else(|| profile_stem(&path)),
            label: string_field(&raw, "label").unwrap_or_else(|| "Custom profile".to_string()),
            api_types,
            api_key,
            raw,
        })
    }

    fn base_url(&self, protocol: WireProtocol) -> String {
        let api_type = api_type_for(protocol);
        self.raw
            .pointer(&format!("/overrides/{api_type}/base_url"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| panic!("profile '{}' has no base_url for {api_type}", self.id))
            .to_string()
    }

    fn model(&self, protocol: WireProtocol) -> String {
        let api_type = api_type_for(protocol);
        self.raw
            .pointer(&format!("/overrides/{api_type}/model"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| panic!("profile '{}' has no model for {api_type}", self.id))
            .to_string()
    }
}

fn translator_for(protocol: WireProtocol) -> Box<dyn WireTranslator> {
    match protocol {
        WireProtocol::OpenAiResponses => Box::new(OpenAiResponsesTranslator),
        WireProtocol::OpenAiChat => Box::new(OpenAiChatTranslator),
        WireProtocol::AnthropicMessages => Box::new(AnthropicMessagesTranslator),
    }
}

fn request_for(protocol: WireProtocol, model: &str, prompt: &str) -> Value {
    match protocol {
        WireProtocol::OpenAiResponses => json!({
            "model": model,
            "input": prompt,
            "max_output_tokens": 32,
            "stream": false
        }),
        WireProtocol::OpenAiChat => json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_completion_tokens": 32,
            "stream": false
        }),
        WireProtocol::AnthropicMessages => json!({
            "model": model,
            "max_tokens": 32,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false
        }),
    }
}

fn post_json(protocol: WireProtocol, endpoint: &str, api_key: &str, body: &Value) -> Result<Value> {
    let body = serde_json::to_string(body).unwrap();
    let mut command = Command::new("curl");
    command
        .arg("-sS")
        .arg("-X")
        .arg("POST")
        .arg(endpoint)
        .arg("-H")
        .arg("content-type: application/json")
        .arg("-d")
        .arg(body)
        .arg("-w")
        .arg("\n__VA_HTTP_STATUS__:%{http_code}");

    match protocol {
        WireProtocol::AnthropicMessages => {
            command
                .arg("-H")
                .arg(format!("x-api-key: {api_key}"))
                .arg("-H")
                .arg(format!("authorization: Bearer {api_key}"))
                .arg("-H")
                .arg("anthropic-version: 2023-06-01");
        }
        WireProtocol::OpenAiResponses | WireProtocol::OpenAiChat => {
            command
                .arg("-H")
                .arg(format!("authorization: Bearer {api_key}"));
        }
    }

    let output = command.output().map_err(|error| {
        va_ai_api_proxy::ApiProxyError::invalid_response(format!("run curl: {error}"))
    })?;
    if !output.status.success() {
        return Err(va_ai_api_proxy::ApiProxyError::invalid_response(format!(
            "curl exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (body, status) = split_curl_status(&stdout)?;
    if !(200..300).contains(&status) {
        return Err(va_ai_api_proxy::ApiProxyError::invalid_response(format!(
            "upstream returned HTTP {status}: {body}"
        )));
    }
    serde_json::from_str(body).map_err(|error| {
        va_ai_api_proxy::ApiProxyError::invalid_response(format!(
            "parse upstream JSON response: {error}; body: {body}"
        ))
    })
}

fn split_curl_status(stdout: &str) -> Result<(&str, u16)> {
    let marker = "\n__VA_HTTP_STATUS__:";
    let (body, status) = stdout.rsplit_once(marker).ok_or_else(|| {
        va_ai_api_proxy::ApiProxyError::invalid_response("curl output missing HTTP status marker")
    })?;
    let status = status.trim().parse::<u16>().map_err(|error| {
        va_ai_api_proxy::ApiProxyError::invalid_response(format!("parse HTTP status: {error}"))
    })?;
    Ok((body, status))
}

fn endpoint_for(protocol: WireProtocol, base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    match protocol {
        WireProtocol::OpenAiResponses => format!("{base_url}/responses"),
        WireProtocol::OpenAiChat => format!("{base_url}/chat/completions"),
        WireProtocol::AnthropicMessages => {
            if base_url.ends_with("/v1") {
                format!("{base_url}/messages")
            } else {
                format!("{base_url}/v1/messages")
            }
        }
    }
}

fn api_type_for(protocol: WireProtocol) -> &'static str {
    match protocol {
        WireProtocol::OpenAiResponses => "openai-responses",
        WireProtocol::OpenAiChat => "openai-chat",
        WireProtocol::AnthropicMessages => "anthropic",
    }
}

fn parse_protocol(value: &str) -> Result<WireProtocol> {
    match value {
        "anthropic" => Ok(WireProtocol::AnthropicMessages),
        other => WireProtocol::from_str(other),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<Option<String>> {
    args.next().map(Some).ok_or_else(|| {
        va_ai_api_proxy::ApiProxyError::invalid_request(format!("{name} requires a value"))
    })
}

fn profiles_dir() -> PathBuf {
    let home = env::var_os("HOME").expect("HOME must be set");
    PathBuf::from(home).join(".vibearound").join("profiles")
}

fn find_custom_profile() -> Result<PathBuf> {
    let profiles_dir = profiles_dir();
    let entries = std::fs::read_dir(&profiles_dir).map_err(|error| {
        va_ai_api_proxy::ApiProxyError::invalid_request(format!(
            "read profiles dir {}: {error}",
            profiles_dir.display()
        ))
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&body) else {
            continue;
        };
        if value.get("provider").and_then(Value::as_str) == Some("custom") {
            return Ok(path);
        }
    }

    Err(va_ai_api_proxy::ApiProxyError::invalid_request(format!(
        "no custom profile found in {}",
        profiles_dir.display()
    )))
}

fn profile_stem(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("profile")
        .to_string()
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn strip_source_raw(request: &mut UniversalRequest) {
    if let Some(source) = &mut request.source {
        source.raw = None;
    }
}

fn print_usage() {
    eprintln!(
        r#"Usage:
  cargo run --example live_smoke -- --profile custom-vvjlcv80 --protocol openai-responses
  cargo run --example live_smoke -- --profile custom-vvjlcv80 --protocol anthropic

Options:
  --profile <id>      VibeAround profile id. Defaults to the first custom profile.
  --protocol <kind>   openai-responses | anthropic | anthropic-messages | openai-chat
  --prompt <text>     Prompt to send. Defaults to a tiny fixed smoke prompt.
  --url <url>         Override full endpoint URL.
  --model <model>     Override model.
"#
    );
}
