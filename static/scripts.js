// Get the textarea and sentiment result elements
const messageInput = document.getElementById('message');
const sentimentResult = document.getElementById('sentiment-result');

// Add an event listener to the textarea to handle new input
messageInput.addEventListener('input', () => {
    // Clear the sentiment result by hiding or emptying the content
    sentimentResult.style.display = 'none'; // Hides the result area
});
