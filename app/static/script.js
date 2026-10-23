/* ArcticDense Interactive Search Engine Logic */
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsContainer = document.getElementById('results-container');
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const telemetryBar = document.getElementById('telemetry-bar');
    const latencyVal = document.getElementById('latency-val');
    const resultCount = document.getElementById('result-count');
    const mrlIndicator = document.getElementById('mrl-indicator');
    const genreSelect = document.getElementById('genre-select');
    const topkSelect = document.getElementById('topk-select');
    const mrlChips = document.querySelectorAll('.pill-opt');
    const suggestionChips = document.querySelectorAll('.chip');

    // Modal Elements
    const modalOverlay = document.getElementById('modal-overlay');
    const modalClose = document.getElementById('modal-close');
    const modalTitle = document.getElementById('modal-title');
    const modalScore = document.getElementById('modal-score');
    const modalGenres = document.getElementById('modal-genres');
    const modalSynopsis = document.getElementById('modal-synopsis');
    const modalBid = document.getElementById('modal-bid');
    const copyBtn = document.getElementById('copy-btn');

    // State
    let currentDim = 768;
    let currentGenre = '';
    let currentTopK = 10;
    let debounceTimer = null;
    let lastQuery = '';

    // Keyboard shortcut '/' to focus
    window.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement !== searchInput) {
            e.preventDefault();
            searchInput.focus();
            searchInput.select();
        } else if (e.key === 'Escape') {
            closeModal();
        }
    });

    // MRL Dimension Selector
    mrlChips.forEach(btn => {
        btn.addEventListener('click', () => {
            mrlChips.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentDim = parseInt(btn.dataset.dim, 10);
            if (searchInput.value.trim()) {
                performSearch();
            }
        });
    });

    // Genre Filter Selector
    genreSelect.addEventListener('change', () => {
        currentGenre = genreSelect.value;
        if (searchInput.value.trim()) {
            performSearch();
        }
    });

    // Top-K Selector
    topkSelect.addEventListener('change', () => {
        currentTopK = parseInt(topkSelect.value, 10);
        if (searchInput.value.trim()) {
            performSearch();
        }
    });

    // Input handlers
    searchInput.addEventListener('input', () => {
        const val = searchInput.value.trim();
        clearBtn.style.display = val ? 'inline-block' : 'none';
        
        clearTimeout(debounceTimer);
        if (val.length >= 3) {
            debounceTimer = setTimeout(() => {
                performSearch();
            }, 350);
        } else if (val.length === 0) {
            resetSearch();
        }
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            clearTimeout(debounceTimer);
            performSearch();
        }
    });

    searchBtn.addEventListener('click', () => {
        performSearch();
    });

    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearBtn.style.display = 'none';
        searchInput.focus();
        resetSearch();
    });

    // Suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            searchInput.value = query;
            clearBtn.style.display = 'inline-block';
            performSearch();
        });
    });

    function resetSearch() {
        resultsContainer.innerHTML = '';
        resultsContainer.appendChild(emptyState);
        emptyState.style.display = 'block';
        telemetryBar.style.display = 'none';
        lastQuery = '';
    }

    // Perform Vector Search API Call
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        // Show loading state
        emptyState.style.display = 'none';
        loadingState.style.display = 'block';
        
        // Remove existing grid while keeping loader
        const existingGrid = document.querySelector('.books-grid');
        if (existingGrid) existingGrid.remove();

        try {
            const params = new URLSearchParams({
                q: query,
                dim: currentDim,
                top_k: currentTopK
            });
            if (currentGenre) {
                params.append('genre', currentGenre);
            }

            const response = await fetch(`/api/search?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`Search request failed with status ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);
            lastQuery = query;
        } catch (error) {
            console.error('Search error:', error);
            loadingState.style.display = 'none';
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <h3>Search failed</h3>
                    <p style="color: #ef4444;">${error.message}. Please check if the model backend is ready.</p>
                </div>
            `;
        }
    }

    // Render results grid
    function renderResults(data) {
        loadingState.style.display = 'none';

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <h3>No matches found</h3>
                    <p>Try broadening your query or removing the genre filter.</p>
                </div>
            `;
            telemetryBar.style.display = 'none';
            return;
        }

        // Update telemetry
        telemetryBar.style.display = 'flex';
        latencyVal.textContent = `${data.latency_ms} ms`;
        resultCount.textContent = `Found ${data.results.length} books in catalog`;
        mrlIndicator.textContent = `${data.mrl_dim}-d Matryoshka Vector`;

        const grid = document.createElement('div');
        grid.className = 'books-grid';

        data.results.forEach((book, index) => {
            const card = document.createElement('div');
            card.className = 'book-card';
            card.style.animation = `modalSlide 0.3s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.04}s backwards`;

            const genresHtml = (book.genres || [])
                .slice(0, 3)
                .map(g => `<span class="genre-pill">${escapeHtml(g)}</span>`)
                .join('');

            card.innerHTML = `
                <div>
                    <div class="book-card-header">
                        <h3 class="book-title">${escapeHtml(book.title)}</h3>
                        <span class="score-badge">${book.score_pct} Match</span>
                    </div>
                    <div class="book-genres">${genresHtml}</div>
                    <p class="book-snippet">${escapeHtml(book.snippet)}</p>
                </div>
                <div class="book-card-footer">
                    <span class="book-id-tag">ID: ${book.book_id}</span>
                    <span class="read-more-link">View Details &rarr;</span>
                </div>
            `;

            card.addEventListener('click', () => {
                openModal(book);
            });

            grid.appendChild(card);
        });

        resultsContainer.innerHTML = '';
        resultsContainer.appendChild(grid);
    }

    // Modal handlers
    function openModal(book) {
        modalTitle.textContent = book.title;
        modalScore.textContent = `${book.score_pct} Semantic Match`;
        modalSynopsis.textContent = book.full_description || 'No detailed synopsis available for this book.';
        modalBid.textContent = `Goodreads Book ID: ${book.book_id}`;

        modalGenres.innerHTML = (book.genres || [])
            .map(g => `<span class="genre-pill">${escapeHtml(g)}</span>`)
            .join('');

        modalOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        copyBtn.onclick = () => {
            const textToCopy = `"${book.title}" [Goodreads ID: ${book.book_id}]\n${book.full_description}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyBtn.textContent = 'Copied!';
                setTimeout(() => { copyBtn.textContent = 'Copy Details'; }, 2000);
            });
        };
    }

    function closeModal() {
        modalOverlay.style.display = 'none';
        document.body.style.overflow = '';
    }

    modalClose.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });

    function escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
});
