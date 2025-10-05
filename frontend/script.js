// Grab DOM elements
const termsModal = document.getElementById('terms-modal');
const resultModal = document.getElementById('result-modal');
const submitBtn = document.getElementById('submit-btn');
const reviewInput = document.getElementById('review-input');
const termsAcceptBtn = document.getElementById('terms-accept-btn');
const termsDeclineBtn = document.getElementById('terms-decline-btn');

const resultTitle = document.getElementById('result-title');
const resultMessage = document.getElementById('result-message');

// Handle Submit Button Click - Show Terms First
submitBtn.addEventListener('click', () => {
    const reviewText = reviewInput.value.trim();

    if (!reviewText) {
        alert("Please enter a review before submitting.");
        return;
    }

    // Show terms and conditions modal
    termsModal.classList.add('show');
});

// Handle Terms Accept
termsAcceptBtn.addEventListener('click', async () => {
    const reviewText = reviewInput.value.trim();
    
    // Close terms modal
    termsModal.classList.remove('show');
    
    // Open result modal immediately
    resultModal.classList.add('show');
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
        const isGenuine = result.prediction.toLowerCase() === "normal";
        
        if (isGenuine) {
            resultTitle.textContent = "Genuine Review";
            resultMessage.innerHTML = `
                <p style="font-size: 1.1rem; margin-bottom: 20px;">This review appears <strong>Normal</strong> and shows typical patterns found in genuine customer feedback.</p>
            `;
        } else {
            resultTitle.textContent = "Suspicious Review";
            resultMessage.innerHTML = `
                <p style="font-size: 1.1rem; margin-bottom: 20px;">This review appears <strong>Suspicious</strong> and shows unusual patterns that may indicate fraudulent activity.</p>
            `;
        }
    } catch (error) {
        console.error("Fetch or processing error:", error);
        resultTitle.textContent = "Error!";
        resultMessage.textContent = "There was an issue connecting to the server. Please try again later.";
    }
});

// Handle Terms Decline
termsDeclineBtn.addEventListener('click', () => {
    termsModal.classList.remove('show');
});

// Close terms modal when clicking outside of the content area
termsModal.addEventListener('click', (event) => {
    if (event.target === termsModal) {
        termsModal.classList.remove('show');
    }
});

// Close result modal when clicking outside of the content area
resultModal.addEventListener('click', (event) => {
    if (event.target === resultModal) {
        resultModal.classList.remove('show');
    }
});