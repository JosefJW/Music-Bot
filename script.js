// script.js

// Initialize an array to store the songs added by the user
let songs = [];

function addSong() {
    const songInput = document.getElementById("song-input");
    const songName = songInput.value.trim();

    // Add song only if it's not empty and isn't already in the list
    if (songName && !songs.includes(songName)) {
        songs.push(songName);
        renderSongBubbles();
        songInput.value = ""; // Clear input field
        getRecommendations(); // Update recommendations
    }
}


// Function to render the song bubbles
function renderSongBubbles() {
    const container = document.getElementById("songs-bubble-container");
    container.innerHTML = ""; // Clear existing bubbles

    // Loop through the songs and create a bubble for each
    songs.forEach(function(song) {
        const bubble = document.createElement("div");
        bubble.className = "song-bubble";
        bubble.textContent = song;

        // Add event listener to remove the song when clicked
        bubble.onclick = function() {
            removeSong(song);
        };

        container.appendChild(bubble);
    });
}

// Function to remove a song from the list and re-render the bubbles
function removeSong(song) {
    songs = songs.filter(function(s) {
        return s !== song;
    });
    renderSongBubbles();
    getRecommendations();
}

// Function to simulate getting song recommendations
function getRecommendations() {
    const recommendationsDiv = document.getElementById("recommendations");
    recommendationsDiv.innerHTML = ""; // Clear previous recommendations

    // Example: Mock data - generate recommendations based on current songs
    const mockRecommendations = songs.map(song => ({
        name: song,
        album: `Album of ${song}`,
        similarity: Math.floor(Math.random() * 100) // Random similarity
    }));

    // Display recommendations
    mockRecommendations.forEach(function(song) {
        const card = document.createElement("div");
        card.className = "recommendation-card";
        card.innerHTML = `
            <h3>${song.name}</h3>
            <p>Album: ${song.album}</p>
            <p>Similarity: ${song.similarity}%</p>
            <button class="thumb-button thumb-up">👍</button>
            <button class="thumb-button thumb-down">👎</button>
        `;

        card.querySelector(".thumb-up").addEventListener("click", function() {
            thumbsUp(card);
        })

        card.querySelector(".thumb-down").addEventListener("click", function() {
            thumbsDown(card);
        })

        recommendationsDiv.appendChild(card);
    });
}

function thumbsUp(card) {
    handleLike(card);
    card.classList.add("flash-green");

    // Remove the class after the animation ends
    card.addEventListener("animationend", function() {
        card.classList.remove("flash-green");
    }, { once: true }); // Ensures the event listener runs only once
}

function thumbsDown(card) {
    handleDislike(card);
    card.classList.add("flash-red");

    // Remove the class after the animation ends
    card.addEventListener("animationend", function() {
        card.classList.remove("flash-red");
    }, { once: true }); // Ensures the event listener runs only once
}

// Update bot based on feedback
function handleLike(card) {}
function handleDislike(card) {}
