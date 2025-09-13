// Grab DOM elements
const modal = document.getElementById('result-modal');
const submitBtn = document.getElementById('submit-btn');
const reviewInput = document.getElementById('review-input');

const resultTitle = document.getElementById('result-title');
const resultMessage = document.getElementById('result-message');

// Handle Submit Button Click
submitBtn.addEventListener('click', async () => {
    const reviewText = reviewInput.value.trim();

    if (!reviewText) {
        alert("Please enter a review before submitting.");
        return;
    }

    // Open modal immediately
    modal.classList.add('show');
    resultTitle.textContent = "Analyzing...";
    resultMessage.textContent = "Please wait while we process your review.";

    try {
        // Send POST request to FastAPI backend
        const response = await fetch("http://127.0.0.1:8000/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ review: reviewText })
        });

        // Convert to JSON
        const data = await response.json();

        // Debug: Check the actual backend response
        console.log("Raw backend response:", data);

        // Decide if response has `result` key or not
        let result;
        if (data.result) {
            // If backend wraps output inside `result`
            result = data.result;
        } else {
            // If backend directly returns the prediction JSON
            result = data;
        }

        // Display prediction in the modal
        resultTitle.textContent = result.prediction; // "Normal" or "Anomalous"
        resultMessage.innerHTML = `
            <strong>Cluster:</strong> ${result.cluster}<br>
            <strong>Distance:</strong> ${result.distance.toFixed(4)}<br>
            <strong>Processed Review:</strong> ${result.processed_text}
        `;
    } catch (error) {
        console.error("Fetch or processing error:", error);
        resultTitle.textContent = "Error!";
        resultMessage.textContent = "There was an issue connecting to the server. Please try again later.";
    }
});

// Close modal when clicking outside of the content area
modal.addEventListener('click', (event) => {
    if (event.target === modal) {
        modal.classList.remove('show');
    }
});
