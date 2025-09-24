const form = document.getElementById('ask-form');
const answers = document.getElementById('answers');
const CTGOV_STUDY_BASE_URL = 'https://clinicaltrials.gov/study/';

const renderCitation = (citation) => {
    const trialUrl = `${CTGOV_STUDY_BASE_URL}${encodeURIComponent(citation.nct_id)}`;
    return `<div class="cite">[<a href="${trialUrl}" target="_blank" rel="noopener noreferrer">${citation.nct_id}</a>] <em>${citation.section}</em>: ${citation.text_snippet}</div>`;
};

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('query').value;
    const nct = document.getElementById('nct').value || null;
    const res = await fetch('/ask/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, nct_id: nct })
    });
    const data = await res.json();
    const block = document.createElement('div');
    block.className = 'card';
    const citationMarkup = data.citations.map(renderCitation).join('');
    block.innerHTML = `<p>${data.answer}</p>${citationMarkup}`;
    answers.prepend(block);
});
