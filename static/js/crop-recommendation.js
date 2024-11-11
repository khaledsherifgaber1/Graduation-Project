// crop-recommendation.js

document.querySelector('.crop-form').addEventListener('submit', function(e) {
    e.preventDefault();

    // Example values for testing - replace this logic with actual recommendation engine
    const recommendedCrop = "Rice";  // Hard-coded for demonstration purposes

    // Display result
    document.getElementById('result').textContent = `Recommended Crop: ${recommendedCrop}`;
});
