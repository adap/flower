export const SUMMARY_PROMPT_CACHE_KEY = 'flowerintelligence-summarizer-prompt';
export const REPLY_CACHE_KEY = 'flowerintelligence-reply-prompt';
export const REMOTE_HANDOFF_CACHE_KEY = 'flowerintelligence-remote-handoff';
export const SUMMARY_CACHE_KEY = 'flowerintelligence-summary';

export const DEFAULT_SUMMARY_PROMPT =
  'Can you write a short summary the following email (your reply should only contain the short summary)?';
export const DEFAULT_REPLY_PROMPT =
  'Can you write a reply to the following email (the reply should start with `<REPLY_BEGIN>` and end with `<REPLY_END>`)?';
export const DEFAULT_REMOTE_HANDOFF = true;

export const NO_SUMMARY_AVAILABLE = 'No summary available.';
export const FAILED_SUMMARY = 'Failed to generate summary.';

export const GEN_SUMMARY_CMD = 'generateSummary';
export const GEN_REPLY_CMD = 'generateReply';
export const REGEN_SUMMARY_CMD = 'regenerateSummary';
export const GET_BANNER_CMD = 'getBannerDetails';
export const GET_CURRENT_MSG_CMD = 'getCurrentMessageId';
export const SAVE_SUMMARY_CMD = 'saveSummary';

export const CURR_EMAIL_CACHE_KEY = 'flowerIntelligenceChatCurrentEmailBody';
