const form = document.getElementById('ask-form');
const answers = document.getElementById('answers');
const askButton = document.getElementById('ask-button');
const status = document.getElementById('status');
const queryInput = document.getElementById('query');
const newChatButton = document.getElementById('new-chat-button');
const themeToggleButton = document.getElementById('theme-toggle');
const suggestionButtons = document.querySelectorAll('[data-query]');
const CTGOV_STUDY_BASE_URL = 'https://clinicaltrials.gov/study/';

const setLoadingState = (isLoading) => {
    askButton.disabled = isLoading;
    askButton.classList.toggle('loading', isLoading);
};

const updateNewChatState = () => {
    if (!newChatButton) {
        return;
    }
    const hasConversation = answers.children.length > 0 || queryInput.value.trim().length > 0;
    newChatButton.disabled = !hasConversation;
};

const renderCitation = (citation) => {
    const trialUrl = `${CTGOV_STUDY_BASE_URL}${encodeURIComponent(citation.nct_id)}`;
    return `<div class="cite">[<a href="${trialUrl}" target="_blank" rel="noopener noreferrer">${citation.nct_id}</a>] <em>${citation.section}</em>: ${citation.text_snippet}</div>`;
};

const renderCitationsSection = (citations) => {
    if (!Array.isArray(citations) || citations.length === 0) {
        return '';
    }
    const summaryLabel = citations.length === 1 ? 'Show 1 citation' : `Show ${citations.length} citations`;
    const citationMarkup = citations.map(renderCitation).join('');
    return `<details class="citations"><summary>${summaryLabel}</summary>${citationMarkup}</details>`;
};

const typeText = (element, text) =>
    new Promise((resolve) => {
        const delay = 20;
        const stepSize = Math.max(1, Math.ceil(text.length / 200));
        let index = 0;

        const typeNext = () => {
            index = Math.min(text.length, index + stepSize);
            element.textContent = text.slice(0, index);

            if (index < text.length) {
                setTimeout(typeNext, delay);
            } else {
                resolve();
            }
        };

        typeNext();
    });

const clearConversation = () => {
    answers.innerHTML = '';
    status.textContent = '';
    form.reset();
    queryInput.focus();
    updateNewChatState();
};

const setTheme = (theme) => {
    document.body.dataset.theme = theme;
    if (themeToggleButton) {
        const isDark = theme === 'dark';
        themeToggleButton.setAttribute('aria-pressed', String(isDark));
        const label = isDark ? 'Light mode' : 'Dark mode';
        const labelSpan = themeToggleButton.querySelector('.toggle-label');
        if (labelSpan) {
            labelSpan.textContent = label;
        }
    }
    window.localStorage.setItem('trialwhisperer-theme', theme);
};

const initializeTheme = () => {
    if (!themeToggleButton) {
        return;
    }
    const savedTheme = window.localStorage.getItem('trialwhisperer-theme');
    if (savedTheme === 'dark' || savedTheme === 'light') {
        setTheme(savedTheme);
    } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setTheme(prefersDark ? 'dark' : 'light');
    }

    themeToggleButton.addEventListener('click', () => {
        const nextTheme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
        setTheme(nextTheme);
    });
};

if (newChatButton) {
    newChatButton.addEventListener('click', clearConversation);
    updateNewChatState();
}

queryInput.addEventListener('input', updateNewChatState);

if (suggestionButtons.length > 0) {
    suggestionButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const suggestion = button.getAttribute('data-query');
            if (!suggestion) {
                return;
            }
            queryInput.value = suggestion;
            queryInput.focus();
            updateNewChatState();
        });
    });
}

initializeTheme();

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) {
        updateNewChatState();
        return;
    }

    setLoadingState(true);
    status.textContent = 'Searching for relevant trials...';

    try {
        const res = await fetch('/ask/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (!res.ok) {
            throw new Error('Request failed');
        }

        const data = await res.json();
        const block = document.createElement('div');
        block.className = 'card';
        const answerParagraph = document.createElement('p');
        block.appendChild(answerParagraph);
        answers.prepend(block);

        const answerText = typeof data.answer === 'string' ? data.answer : String(data.answer ?? '');

        await typeText(answerParagraph, answerText);

        const citationsSection = renderCitationsSection(data.citations);
        if (citationsSection) {
            block.insertAdjacentHTML('beforeend', citationsSection);
        }

        status.textContent = '';
    } catch (error) {
        status.textContent = 'Something went wrong. Please try again.';
        console.error(error);
    } finally {
        setLoadingState(false);
        updateNewChatState();
    }
});
