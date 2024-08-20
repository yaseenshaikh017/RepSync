document.addEventListener("DOMContentLoaded", function () {
    const resetButton = document.getElementById("reset-button");
    
    if (resetButton) {
        resetButton.addEventListener("click", function () {
            fetch('/reset_counter_overheadshoulderpress', {
                method: 'POST'
            }).then(response => {
                if (response.ok) {
                    document.getElementById("reps-count").textContent = "0";
                    document.getElementById("highscore-count").textContent = "0";
                }
            });
        });
    }

    // Fetch updates for reps and high score regularly
    setInterval(function () {
        fetch('/update_data_overheadshoulderpress')
            .then(response => response.json())
            .then(data => {
                console.log(data);  // Log the data to check if it's correct
                document.getElementById("reps-count").textContent = data.counter;
                document.getElementById("highscore-count").textContent = data.high_score;
            });
    }, 1000); // Adjust the interval as needed
});
