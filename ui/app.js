const form = document.getElementById('ask-form');
const answers = document.getElementById('answers');

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
    block.innerHTML = `<p>${data.answer}</p>` +
    data.citations.map(c => `<div class="cite">[${c.nct_id}] <em>${c.section}</em>: ${c.text_snippet}</div>`).join('');
    answers.prepend(block);
});