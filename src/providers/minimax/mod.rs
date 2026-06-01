mod request;
mod think_tags;

use crate::UniversalEvent;

use self::think_tags::MiniMaxThinkTagSplitter;

#[derive(Debug, Clone, Default)]
pub struct MiniMaxBridgeAdapter {
    think_tags: MiniMaxThinkTagSplitter,
}

impl MiniMaxBridgeAdapter {
    pub fn prepare_chat_request(&mut self, chat_request: &mut serde_json::Value) {
        request::prepare_chat_request(chat_request);
    }

    pub fn transform_upstream_events(&mut self, events: &mut Vec<UniversalEvent>) {
        self.think_tags.transform(events);
    }
}

#[cfg(test)]
mod tests;
