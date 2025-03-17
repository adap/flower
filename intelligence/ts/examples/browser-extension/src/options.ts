import { storage } from 'webextension-polyfill';
import {
  DEFAULT_REPLY_PROMPT,
  DEFAULT_SUMMARY_PROMPT,
  REPLY_CACHE_KEY,
  SUMMARY_PROMPT_CACHE_KEY,
} from './constants';
import { TOGGLE_ON_ICON, TOGGLE_OFF_ICON } from './icons';

const toggleSettingDisplay = (
  isToggled: boolean,
  toggleButton: HTMLElement,
  settingElement?: HTMLElement
) => {
  if (settingElement) {
    settingElement.style.display = isToggled ? 'block' : 'none';
  }
  toggleButton.innerHTML = isToggled ? TOGGLE_ON_ICON : TOGGLE_OFF_ICON;

  if (isToggled) {
    if (!toggleButton.classList.contains('toggle-on')) {
      toggleButton.classList.add('toggle-on');
    }
  } else {
    toggleButton.classList.remove('toggle-on');
  }
};

function getElement(selector: string): HTMLElement {
  const element = document.querySelector(selector);
  if (!element) {
    throw new Error(`Element with selector "${selector}" not found.`);
  }
  return element as HTMLElement;
}

const elements = {
  summarizePromptInput: getElement('#summarizerPrompt') as HTMLTextAreaElement,
  replyPromptInput: getElement('#replyPrompt') as HTMLTextAreaElement,
  resetReplyButton: getElement('#defaultReply') as HTMLDivElement,
  resetSummaryButton: getElement('#defaultSummary') as HTMLDivElement,
  resetReplyContainer: getElement('#defaultReplyContainer') as HTMLDivElement,
  resetSummaryContainer: getElement('#defaultSummaryContainer') as HTMLDivElement,
  summarySettingContainer: getElement('#summarySettingContainer') as HTMLDivElement,
  summarySetting: getElement('#summarySetting') as HTMLDivElement,
  replySettingContainer: getElement('#replySettingContainer') as HTMLDivElement,
  replySetting: getElement('#replySetting') as HTMLDivElement,
  buttonContainer: getElement('.buttonContainer') as HTMLDivElement,
  settingsButton: getElement('.saveButton') as HTMLDivElement,
  summaryToggleButton: getElement('#summaryToggle') as HTMLDivElement,
  replyToggleButton: getElement('#replyToggle') as HTMLDivElement,
  remoteToggleButton: getElement('#remoteToggle') as HTMLDivElement,
  globalSettingContainer: getElement('#globalSettingContainer') as HTMLDivElement,
  toggleAdvancedButton: getElement('.toggleAdvancedButton') as HTMLDivElement,
  advancedSection: getElement('#advancedSection') as HTMLDivElement,
};

function toggleAdvancedSection() {
  const isVisible = elements.advancedSection.classList.toggle('visible');
  elements.toggleAdvancedButton.textContent = isVisible
    ? '▼ Advanced Settings'
    : '▶ Advanced Settings';
  elements.toggleAdvancedButton.style.marginBottom = isVisible ? '0px' : '16px';
}

const primaryGradient = 'linear-gradient(to right bottom, #0070ff, #9747ff)';
const primaryBlue = '#0070ff';
const primaryGrey = '#b7b7bf';

const updateButtonState = (isModified: boolean) => {
  elements.buttonContainer.style.background = isModified ? primaryGradient : primaryGrey;
  elements.buttonContainer.style.cursor = isModified ? 'pointer' : 'default';

  elements.settingsButton.style.pointerEvents = isModified ? 'auto' : 'none';
};

const updateResetContainerState = (isModified: boolean, resetContainer: HTMLElement) => {
  resetContainer.style.backgroundColor = isModified ? primaryBlue : primaryGrey;
};

let cachedSummaryPrompt = DEFAULT_SUMMARY_PROMPT;
let cachedReplyPrompt = DEFAULT_REPLY_PROMPT;

async function loadPromptValues() {
  const summaryCacheRes = await storage.local.get(SUMMARY_PROMPT_CACHE_KEY);
  const replyCacheRes = await storage.local.get(REPLY_CACHE_KEY);

  let shouldSaveDefaults = false;

  if (!(SUMMARY_PROMPT_CACHE_KEY in summaryCacheRes)) {
    cachedSummaryPrompt = DEFAULT_SUMMARY_PROMPT;
    shouldSaveDefaults = true;
  } else {
    cachedSummaryPrompt = String(summaryCacheRes[SUMMARY_PROMPT_CACHE_KEY]);
  }

  if (!(REPLY_CACHE_KEY in replyCacheRes)) {
    cachedReplyPrompt = DEFAULT_REPLY_PROMPT;
    shouldSaveDefaults = true;
  } else {
    cachedReplyPrompt = String(replyCacheRes[REPLY_CACHE_KEY]);
  }

  if (shouldSaveDefaults) {
    await storage.local.set({
      [SUMMARY_PROMPT_CACHE_KEY]: cachedSummaryPrompt,
      [REPLY_CACHE_KEY]: cachedReplyPrompt,
    });
  }

  elements.summarizePromptInput.value = String(cachedSummaryPrompt);
  elements.replyPromptInput.value = String(cachedReplyPrompt);

  const isSummaryModified = cachedSummaryPrompt !== DEFAULT_SUMMARY_PROMPT;
  const isReplyModified = cachedReplyPrompt !== DEFAULT_REPLY_PROMPT;

  toggleSettingDisplay(isSummaryModified, elements.summaryToggleButton, elements.summarySetting);
  toggleSettingDisplay(isReplyModified, elements.replyToggleButton, elements.replySetting);

  handleInputChange(isSummaryModified, isReplyModified);
}

const handleInputChange = (isSummaryModified: boolean, isReplyModified: boolean) => {
  const isSummaryChanged = elements.summarizePromptInput.value !== cachedSummaryPrompt;
  const isReplyChanged = elements.replyPromptInput.value !== cachedReplyPrompt;

  const anyModified = isSummaryChanged || isReplyChanged;
  updateButtonState(anyModified);

  updateResetContainerState(isSummaryModified, elements.resetSummaryContainer);
  updateResetContainerState(isReplyModified, elements.resetReplyContainer);
};

const attachEventListeners = () => {
  elements.toggleAdvancedButton.addEventListener('click', toggleAdvancedSection);

  elements.replyPromptInput.addEventListener('input', () => {
    handleInputChange(
      elements.summarizePromptInput.value !== DEFAULT_SUMMARY_PROMPT,
      elements.replyPromptInput.value !== DEFAULT_REPLY_PROMPT
    );
  });

  elements.summarizePromptInput.addEventListener('input', () => {
    handleInputChange(
      elements.summarizePromptInput.value !== DEFAULT_SUMMARY_PROMPT,
      elements.replyPromptInput.value !== DEFAULT_REPLY_PROMPT
    );
  });

  elements.globalSettingContainer.addEventListener('click', () => {
    const isToggled = !elements.remoteToggleButton.classList.contains('toggle-on');
    toggleSettingDisplay(isToggled, elements.remoteToggleButton);
    handleInputChange(
      elements.summarizePromptInput.value !== DEFAULT_SUMMARY_PROMPT,
      elements.replyPromptInput.value !== DEFAULT_REPLY_PROMPT
    );
  });

  elements.summarySettingContainer.addEventListener('click', () => {
    const isToggled = elements.summarySetting.style.display === 'none';
    toggleSettingDisplay(isToggled, elements.summaryToggleButton, elements.summarySetting);
  });

  elements.replySettingContainer.addEventListener('click', () => {
    const isToggled = elements.replySetting.style.display === 'none';
    toggleSettingDisplay(isToggled, elements.replyToggleButton, elements.replySetting);
  });

  elements.settingsButton.addEventListener(
    'click',
    () =>
      void (async () => {
        await storage.local.set({
          [SUMMARY_PROMPT_CACHE_KEY]: elements.summarizePromptInput.value,
          [REPLY_CACHE_KEY]: elements.replyPromptInput.value,
        });

        cachedSummaryPrompt = elements.summarizePromptInput.value;
        cachedReplyPrompt = elements.replyPromptInput.value;

        updateButtonState(false);
        elements.settingsButton.innerText = 'Saved!';
        setTimeout(() => (elements.settingsButton.innerText = 'Save settings'), 1000);
      })()
  );

  elements.resetSummaryButton.addEventListener('click', () => {
    elements.summarizePromptInput.value = DEFAULT_SUMMARY_PROMPT;
    elements.summarizePromptInput.dispatchEvent(new Event('input'));

    handleInputChange(
      elements.summarizePromptInput.value !== DEFAULT_SUMMARY_PROMPT,
      elements.replyPromptInput.value !== DEFAULT_REPLY_PROMPT
    );
  });

  elements.resetReplyButton.addEventListener('click', () => {
    elements.replyPromptInput.value = DEFAULT_REPLY_PROMPT;
    elements.replyPromptInput.dispatchEvent(new Event('input'));

    handleInputChange(
      elements.summarizePromptInput.value !== DEFAULT_SUMMARY_PROMPT,
      elements.replyPromptInput.value !== DEFAULT_REPLY_PROMPT
    );
  });
};

async function init() {
  await loadPromptValues();
  attachEventListeners();
}

init().catch((error: unknown) => {
  console.error('Error initializing:', error);
});
