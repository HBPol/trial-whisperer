const form = document.getElementById('ask-form');
const answers = document.getElementById('answers');
const askButton = document.getElementById('ask-button');
const status = document.getElementById('status');
const CTGOV_STUDY_BASE_URL = 'https://clinicaltrials.gov/study/';

const setLoadingState = (isLoading) => {
    askButton.disabled = isLoading;
    askButton.classList.toggle('loading', isLoading);
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

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryInput = document.getElementById('query');
    const query = queryInput.value.trim();
    if (!query) {
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
        const citationsSection = renderCitationsSection(data.citations);
        block.innerHTML = `<p>${data.answer}</p>${citationsSection}`;
        answers.prepend(block);
        status.textContent = '';
    } catch (error) {
        status.textContent = 'Something went wrong. Please try again.';
        console.error(error);
    } finally {
        setLoadingState(false);
    }
});
