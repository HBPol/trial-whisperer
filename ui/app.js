const form = document.getElementById('ask-form');
const answers = document.getElementById('answers');
const askButton = document.getElementById('ask-button');
const status = document.getElementById('status');
const queryInput = document.getElementById('query');
const newChatButton = document.getElementById('new-chat-button');
const suggestionButtons = document.querySelectorAll('[data-query]');
const ingestionSummaryText = document.getElementById('ingestion-summary-text');
const ingestionInfoOpenButton = document.getElementById('ingestion-info-open');
const ingestionInfoDialog = document.getElementById('ingestion-info-dialog');
const ingestionInfoList = document.getElementById('ingestion-info-list');
const ingestionInfoCloseButton = document.getElementById('ingestion-info-close');
const ingestionInfoDismissButton = document.getElementById('ingestion-info-dismiss');
const ingestionInfoUpdated = document.getElementById('ingestion-info-updated');
const ingestionInfoIntro = document.getElementById('ingestion-info-intro');
const supportsDialog = ingestionInfoDialog && typeof ingestionInfoDialog.showModal === 'function';
let hasAutoShownIngestionDialog = false;
const CTGOV_STUDY_BASE_URL = 'https://clinicaltrials.gov/study/';
const NCT_ID_PATTERN = /\bNCT\d{8}\b/i;

let lastNctId = null;

const numberFormatter = new Intl.NumberFormat();

const closeIngestionDialog = () => {
    if (ingestionInfoDialog && ingestionInfoDialog.open) {
        ingestionInfoDialog.close();
    }
};

const openIngestionDialog = ({ auto = false } = {}) => {
    if (!supportsDialog || !ingestionInfoDialog) {
        return;
    }
    if (!ingestionInfoDialog.open) {
        ingestionInfoDialog.showModal();
    }
    if (auto || !hasAutoShownIngestionDialog) {
        hasAutoShownIngestionDialog = true;
    }
};

const maybeAutoShowIngestionDialog = () => {
    if (!supportsDialog || hasAutoShownIngestionDialog) {
        return;
    }
    openIngestionDialog({ auto: true });
};

if (ingestionInfoDialog && supportsDialog) {
    ingestionInfoDialog.addEventListener('cancel', (event) => {
        event.preventDefault();
        closeIngestionDialog();
    });
}

if (ingestionInfoOpenButton) {
    if (supportsDialog) {
        ingestionInfoOpenButton.hidden = false;
        ingestionInfoOpenButton.addEventListener('click', () => openIngestionDialog());
    } else {
        ingestionInfoOpenButton.hidden = true;
    }
}

if (ingestionInfoCloseButton) {
    ingestionInfoCloseButton.addEventListener('click', closeIngestionDialog);
}

if (ingestionInfoDismissButton) {
    ingestionInfoDismissButton.addEventListener('click', closeIngestionDialog);
}

const updateIngestionSummary = (summary) => {
    if (!summary || typeof summary !== 'object') {
        if (ingestionSummaryText) {
            ingestionSummaryText.textContent = 'TrialWhisperer demo: dataset details unavailable.';
        }
        if (ingestionInfoList) {
            ingestionInfoList.innerHTML = '';
            const li = document.createElement('li');
            li.textContent = 'Dataset metadata could not be loaded.';
            ingestionInfoList.appendChild(li);
        }
        if (ingestionInfoUpdated) {
            ingestionInfoUpdated.hidden = true;
            ingestionInfoUpdated.textContent = '';
        }
        return;
    }

    const studyCount = typeof summary.study_count === 'number' ? summary.study_count : null;
    const queryTerms = Array.isArray(summary.query_terms)
        ? summary.query_terms.filter((item) => typeof item === 'string' && item.trim().length > 0)
        : [];
    const filters = summary.filters && typeof summary.filters === 'object' ? summary.filters : {};
    const maxStudies = typeof summary.max_studies === 'number' ? summary.max_studies : null;

    if (ingestionInfoList) {
        ingestionInfoList.innerHTML = '';
        const addItem = (label, value) => {
            if (!value) {
                return;
            }
            const item = document.createElement('li');
            const strong = document.createElement('strong');
            strong.textContent = `${label}: `;
            item.appendChild(strong);
            item.appendChild(document.createTextNode(value));
            ingestionInfoList.appendChild(item);
        };

        if (typeof studyCount === 'number') {
            addItem('Studies indexed', numberFormatter.format(studyCount));
        }
        if (queryTerms.length > 0) {
            addItem('Query terms', queryTerms.join(', '));
        }

        const filterEntries = Object.entries(filters).filter(([, value]) => {
            if (Array.isArray(value)) {
                return value.length > 0;
            }
            return Boolean(value);
        });
        if (filterEntries.length > 0) {
            const filterDescriptions = filterEntries.map(([key, value]) => {
                if (Array.isArray(value)) {
                    return `${key} = ${value.join(', ')}`;
                }
                return `${key} = ${value}`;
            });
            addItem('Filters', filterDescriptions.join('; '));
        }

        if (maxStudies !== null) {
            addItem('Maximum studies requested', numberFormatter.format(maxStudies));
        }
    }

    if (ingestionSummaryText) {
        const parts = [];
        if (typeof studyCount === 'number') {
            parts.push(
                `${numberFormatter.format(studyCount)} ${studyCount === 1 ? 'study' : 'studies'}`,
            );
        }
        if (queryTerms.length > 0) {
            parts.push(`query: ${queryTerms.join(', ')}`);
        }
        if (parts.length > 0) {
            ingestionSummaryText.textContent = `Demo dataset – ${parts.join(' • ')}`;
        } else {
            ingestionSummaryText.textContent = 'TrialWhisperer demo: dataset details unavailable.';
        }
    }

    if (ingestionInfoIntro) {
        if (queryTerms.length > 0) {
            const label = queryTerms.length > 1 ? 'query terms' : 'query term';
            ingestionInfoIntro.textContent =
                `TrialWhisperer is running in demo mode with a limited dataset from ClinicalTrials.gov. It only knows about studies matching the ${label} ${queryTerms.join(', ')}.`;
        } else {
            ingestionInfoIntro.textContent =
                'TrialWhisperer is running in demo mode with a limited dataset. The assistant only knows about the studies described below.';
        }
    }

    if (ingestionInfoUpdated) {
        const timestamp = typeof summary.last_updated === 'string' ? summary.last_updated : null;
        if (timestamp) {
            const parsed = new Date(timestamp);
            if (!Number.isNaN(parsed.getTime())) {
                ingestionInfoUpdated.textContent = `Dataset last updated ${parsed.toLocaleString()}`;
                ingestionInfoUpdated.hidden = false;
            } else {
                ingestionInfoUpdated.hidden = true;
            }
        } else {
            ingestionInfoUpdated.hidden = true;
        }
    }
};

const fetchIngestionSummary = async () => {
    if (!ingestionSummaryText) {
        return;
    }

    try {
        const response = await fetch('/metadata/ingestion-summary');
        if (!response.ok) {
            throw new Error(`Unexpected response: ${response.status}`);
        }
        const payload = await response.json();
        updateIngestionSummary(payload);
        maybeAutoShowIngestionDialog();
    } catch (error) {
        updateIngestionSummary(null);
        if (supportsDialog) {
            maybeAutoShowIngestionDialog();
        }
        console.error('Failed to load ingestion summary', error);
    }
};

fetchIngestionSummary();

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

const extractNctId = (text) => {
    if (!text) {
        return null;
    }
    const match = text.match(NCT_ID_PATTERN);
    return match ? match[0].toUpperCase() : null;
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
    lastNctId = null;
    form.reset();
    queryInput.focus();
    updateNewChatState();
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

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) {
        updateNewChatState();
        return;
    }

    const explicitNctId = extractNctId(query);
    const effectiveNctId = explicitNctId || lastNctId;

    if (!effectiveNctId) {
        status.textContent = 'Please provide a ClinicalTrials.gov NCT ID to start a new chat.';
        queryInput.focus();
        updateNewChatState();
        return;
    }

    setLoadingState(true);
    status.textContent = 'Searching...';

    try {
        const res = await fetch('/ask/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, nct_id: effectiveNctId })
        });

        let data = null;
        try {
            data = await res.json();
        } catch (parseError) {
            if (res.ok) {
                throw parseError;
            }
        }

        if (!res.ok) {
            const message =
                data && typeof data.detail === 'string'
                    ? data.detail
                    : 'Something went wrong. Please try again.';
            status.textContent = message;
            return;
        }

        if (!data || typeof data !== 'object') {
            status.textContent = 'Unexpected response. Please try again.';
            return;
        }

        if (typeof data.nct_id === 'string' && data.nct_id.trim().length > 0) {
            lastNctId = data.nct_id.trim().toUpperCase();
        } else if (explicitNctId) {
            lastNctId = explicitNctId;
        } else if (effectiveNctId) {
            lastNctId = effectiveNctId;
        }

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
        if (!status.textContent) {
            status.textContent = 'Something went wrong. Please try again.';
        }
        console.error(error);
    } finally {
        setLoadingState(false);
        updateNewChatState();
    }
});
