/*
Program Documentation

Program Title:
script.js – Client-Side Logic for USAD Review Anomaly Detection Tool

Programmers:
Cristel Jane Baquing, Angelica Jean Evangelista, James Tristan Landa, Kharl Chester Velasco

Where the Program Fits in the General System Design:
This JavaScript file provides all client-side logic for the USAD system. It connects the HTML page to the backend model, handles user interactions,
runs client-side feature analysis, updates the results panel dynamically, controls modals, toggles technical details, handles glossary controls, and
manages layout synchronization. It acts as the functionality bridge between the interface (index.html) and the anomaly detection API.

Date Written and Revised:
Original version: October 14, 2025
Last revised: November 20, 2025

Purpose:
The purpose of this script is to:
• Handle the review submission process.
• Show and manage the Terms & Conditions modal.
• Send user review data to the backend API (/api/predict).
• Receive prediction results and update the result panel.
• Analyze features client-side to produce user-friendly explanations.
• Display warnings, normal values, anomaly details, and confidence levels.
• Control glossary toggling and technical detail sections.
• Sync the heights between the input box and result panel for alignment.
• Manage UI interactivity, scroll behavior, and tool responsiveness.

Data Structures, Algorithms, and Control:

• Data Structures:
  - DOM references to UI components (buttons, textareas, panels, modals).
  - FEATURE_PROFILES: thresholds for normal and suspicious patterns.
  - FEATURE_DESCRIPTIONS: text explanations for each metric.
  - FEATURE_LABELS: friendly UI names for technical fields.
  - Objects storing prediction results, feature values, and analysis output.

• Algorithms:
  - analyzeFeatures(): evaluates features and returns detailed warnings,
    severity categories, and insights.
  - calculateFeatureContributions(): computes deviation from expected
    thresholds and ranks features contributing most to anomaly score.
  - syncResultPanelHeight(): ensures layout uniformity across tool sections.
  - API request logic using fetch() to communicate with backend prediction API.

• Control:
  - Event listeners handle:
        • Review submission
        • Terms acceptance/decline
        • Modal open/close
        • Screen resizing for responsive UI
        • Glossary visibility and detail toggling
  - Dynamic HTML injection updates the result panel with tables, badges,
    progress bars, tooltips, and structured warnings.
  - Smooth scrolling and visual cues highlight glossary elements for users.
  - Error handling catches server issues and displays fallback messages.
*/


// Grab DOM elements
const termsModal = document.getElementById('terms-modal');
const invalidModal = document.getElementById('invalid-modal');
const submitBtn = document.getElementById('submit-btn');
const reviewInput = document.getElementById('review-input');
const termsAcceptBtn = document.getElementById('terms-accept-btn');
const termsDeclineBtn = document.getElementById('terms-decline-btn');
const invalidOkBtn = document.getElementById('invalid-ok-btn');
const invalidMessage = document.getElementById('invalid-message');

const resultTitle = document.getElementById('result-title');
const resultMessage = document.getElementById('result-message');
const wordCount = document.getElementById('word-count');

// Word count limit
const MAX_WORDS = 200;

// Function to count words in text
function countWords(text) {
    if (!text || text.trim().length === 0) {
        return 0;
    }
    // Split by whitespace and filter out empty strings
    const words = text.trim().split(/\s+/).filter(word => word.length > 0);
    return words.length;
}

// Update word counter
function updateWordCounter() {
    if (!reviewInput || !wordCount) return;
    
    const text = reviewInput.value;
    const wordCountValue = countWords(text);
    
    // Update display
    wordCount.textContent = `${wordCountValue} / ${MAX_WORDS} words`;
    
    // Change color if approaching or exceeding limit
    if (wordCountValue > MAX_WORDS) {
        wordCount.style.color = '#d32f2f'; // Red
        wordCount.style.fontWeight = 'bold';
    } else if (wordCountValue > MAX_WORDS * 0.9) {
        wordCount.style.color = '#f57c00'; // Orange
        wordCount.style.fontWeight = '600';
    } else {
        wordCount.style.color = ''; // Default color
        wordCount.style.fontWeight = '';
    }
}

// Add event listener to update word counter as user types
if (reviewInput) {
    let previousWordCount = 0;
    let lastPopupTime = 0;
    const POPUP_COOLDOWN = 1500; // Show popup at most once every 1.5 seconds to avoid spam
    
    reviewInput.addEventListener('input', (e) => {
        const text = e.target.value;
        const wordCountValue = countWords(text);
        
        // Always truncate if exceeds limit
        if (wordCountValue > MAX_WORDS) {
            // Find the position of the 200th word
            const words = text.trim().split(/\s+/);
            if (words.length > MAX_WORDS) {
                const truncatedWords = words.slice(0, MAX_WORDS);
                const truncatedText = truncatedWords.join(' ');
                
                // Set cursor position to end of truncated text
                const cursorPos = truncatedText.length;
                e.target.value = truncatedText;
                e.target.setSelectionRange(cursorPos, cursorPos);
                
                // Show popup if user is trying to exceed limit (with cooldown to avoid spam)
                const now = Date.now();
                if (now - lastPopupTime > POPUP_COOLDOWN) {
                    showInvalidModal("Invalid data entry");
                    lastPopupTime = now;
                }
            }
        }
        
        previousWordCount = wordCountValue;
        updateWordCounter();
    });
    
    reviewInput.addEventListener('paste', (e) => {
        // Use setTimeout to catch paste event after content is inserted
        setTimeout(() => {
            const text = reviewInput.value;
            const wordCountValue = countWords(text);
            
            // Truncate if exceeds limit
            if (wordCountValue > MAX_WORDS) {
                const words = text.trim().split(/\s+/);
                if (words.length > MAX_WORDS) {
                    const truncatedWords = words.slice(0, MAX_WORDS);
                    reviewInput.value = truncatedWords.join(' ');
                    
                    // Always show popup on paste if it exceeds limit
                    showInvalidModal("Invalid data entry");
                }
            }
            
            updateWordCounter();
        }, 0);
    });
    
    // Prevent keydown events that would add more words when already at limit
    reviewInput.addEventListener('keydown', (e) => {
        const text = reviewInput.value;
        const wordCountValue = countWords(text);
        
        // If already at or above limit, prevent space and other word-separating characters
        if (wordCountValue >= MAX_WORDS) {
            // Allow backspace, delete, arrow keys, etc.
            const allowedKeys = [
                'Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
                'Home', 'End', 'Tab', 'Enter', 'Escape'
            ];
            
            // Allow Ctrl/Cmd combinations (for copy, paste, select all, etc.)
            if (e.ctrlKey || e.metaKey) {
                return true; // Allow
            }
            
            // Prevent space and other characters that would add new words
            if (!allowedKeys.includes(e.key)) {
                // Check if this key would add a new word
                const testText = text + e.key;
                const testWordCount = countWords(testText);
                
                if (testWordCount > MAX_WORDS) {
                    e.preventDefault();
                    const now = Date.now();
                    if (now - lastPopupTime > POPUP_COOLDOWN) {
                        showInvalidModal("Invalid data entry");
                        lastPopupTime = now;
                    }
                    return false;
                }
            }
        }
    });
    
    // Initial update
    previousWordCount = countWords(reviewInput.value);
    updateWordCounter();
}

function syncResultPanelHeight() {
    const leftPanel = document.querySelector('.tool-box.left');
    const rightPanel = document.querySelector('.sticky-results .result-panel');
    if (!leftPanel || !rightPanel) return;
    const leftHeight = leftPanel.getBoundingClientRect().height;
    rightPanel.style.height = `${leftHeight}px`;
    rightPanel.style.maxHeight = `${leftHeight}px`;
    rightPanel.style.overflowY = 'auto';
}

// Run on load and resize
window.addEventListener('load', syncResultPanelHeight);
window.addEventListener('resize', syncResultPanelHeight);

// Define feature baselines and thresholds based on typical patterns
const FEATURE_PROFILES = {
    normal: {
        review_length: { min: 10, max: 200, optimal: 50 },
        lexical_diversity: { min: 0.5, max: 0.95, optimal: 0.7 },
        avg_word_length: { min: 3.5, max: 6.5, optimal: 4.5 },
        sentiment_polarity: { min: -0.3, max: 0.7, optimal: 0.3 },
        sentiment_subjectivity: { min: 0.3, max: 0.8, optimal: 0.5 },
        word_entropy: { min: 2.0, max: 5.5, optimal: 3.5 },
        repetition_ratio: { min: 0, max: 0.25, optimal: 0.1 },
        exclamation_count: { min: 0, max: 2, optimal: 0 },
        question_count: { min: 0, max: 2, optimal: 0 },
        capital_ratio: { min: 0, max: 0.15, optimal: 0.05 },
        punctuation_density: { min: 0.05, max: 0.25, optimal: 0.15 }
    },
    suspicious: {
        // Patterns that indicate suspicious reviews
        high_excitement: { exclamation_count: 5, capital_ratio: 0.3 },
        low_diversity: { lexical_diversity: 0.3, repetition_ratio: 0.5 },
        extreme_sentiment: { sentiment_polarity: [0.9, -0.8] },
        too_short: { review_length: 3 },
        low_entropy: { word_entropy: 1.0 }
    }
};

// Descriptions for each feature for non-technical users
const FEATURE_DESCRIPTIONS = {
    review_length: "How many words are in the review. Very short reviews can look suspicious.",
    lexical_diversity: "How varied the words are. More unique words usually mean a more natural review.",
    avg_word_length: "Average number of letters per word. Extremely short or long words can be unusual.",
    sentiment_polarity: "Overall positivity or negativity of the review. Extremely high or low can be unnatural.",
    sentiment_subjectivity: "How opinion-based the review is. Very subjective without details can be suspicious.",
    word_entropy: "How complex or varied the wording is. Low complexity may indicate template-like text.",
    repetition_ratio: "How often words are repeated. High repetition is common in fake reviews.",
    exclamation_count: "How many exclamation marks are used. Too many can be a red flag.",
    question_count: "How many question marks are used. Unusual punctuation patterns can be suspicious.",
    capital_ratio: "How much of the text is CAPITALIZED. Excessive caps can look promotional.",
    punctuation_density: "How dense the punctuation is. Unusual patterns can be a signal."
};

// Friendly labels for non-technical users
const FEATURE_LABELS = {
    review_length: "Length of review",
    lexical_diversity: "Word variety",
    avg_word_length: "Average word length",
    sentiment_polarity: "Overall tone",
    sentiment_subjectivity: "Opinion level",
    word_entropy: "Language complexity",
    repetition_ratio: "Word repetition",
    exclamation_count: "Exclamation marks",
    question_count: "Question marks",
    capital_ratio: "Capital letter usage",
    punctuation_density: "Punctuation density"
};

// Get API base URL - works for both local and deployed environments
const API_BASE_URL = window.location.origin;

// Load data-driven basis from backend if available and override FEATURE_PROFILES.normal
async function loadFeatureBasis() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/feature-basis`);
        if (!res.ok) return;
        const data = await res.json();
        if (data && data.normal) {
            FEATURE_PROFILES.normal = { ...FEATURE_PROFILES.normal, ...data.normal };
        }
    } catch (_) {
        // Silently ignore if backend not available
    }
}

loadFeatureBasis();

// Analyze features and generate specific reasons
function analyzeFeatures(features, prediction) {
    const reasons = [];
    const warnings = [];
    const profile = FEATURE_PROFILES.normal;
    
    // Review Length Analysis
    const length = features.review_length || 0;
    if (length < profile.review_length.min) {
        reasons.push({
            severity: 'high',
            feature: 'Review Length',
            message: `Very short review (${length} words vs. typical 10+ words)`,
            detail: 'Fake reviews are often brief and lack detailed personal experience'
        });
    } else if (length > profile.review_length.max) {
        warnings.push({
            severity: 'low',
            feature: 'Review Length',
            message: `Unusually long review (${length} words)`,
            detail: 'While not necessarily fake, extremely long reviews can be suspicious'
        });
    }
    
    // Lexical Diversity Analysis
    const diversity = features.lexical_diversity || 0;
    if (diversity < profile.lexical_diversity.min) {
        reasons.push({
            severity: 'high',
            feature: 'Lexical Diversity',
            message: `Low vocabulary variety (${(diversity * 100).toFixed(0)}% unique words)`,
            detail: 'Fake reviews often repeat the same words instead of using varied language'
        });
    }
    
    // Word Entropy Analysis
    const entropy = features.word_entropy || 0;
    if (entropy < profile.word_entropy.min) {
        reasons.push({
            severity: 'medium',
            feature: 'Word Entropy',
            message: `Low linguistic complexity (entropy: ${entropy.toFixed(2)})`,
            detail: 'Genuine reviews typically show more varied word usage patterns'
        });
    }
    
    // Repetition Ratio Analysis
    const repetition = features.repetition_ratio || 0;
    if (repetition > profile.repetition_ratio.max) {
        reasons.push({
            severity: 'high',
            feature: 'Repetition',
            message: `High word repetition (${(repetition * 100).toFixed(0)}% repeated words)`,
            detail: 'Excessive repetition suggests automated or template-based content'
        });
    }
    
    // Sentiment Analysis
    const polarity = features.sentiment_polarity || 0;
    if (Math.abs(polarity) > 0.85) {
        reasons.push({
            severity: 'medium',
            feature: 'Sentiment',
            message: `Extreme ${polarity > 0 ? 'positive' : 'negative'} sentiment (${polarity.toFixed(2)})`,
            detail: 'Fake reviews often express unnaturally extreme emotions'
        });
    }
    
    // Exclamation Analysis
    const exclamations = features.exclamation_count || 0;
    if (exclamations > profile.exclamation_count.max) {
        reasons.push({
            severity: 'medium',
            feature: 'Punctuation',
            message: `Excessive exclamation marks (${exclamations} found)`,
            detail: 'Fake reviews often overuse exclamation marks to appear enthusiastic'
        });
    }
    
    // Capital Letters Analysis
    const capitals = features.capital_ratio || 0;
    if (capitals > profile.capital_ratio.max) {
        reasons.push({
            severity: 'medium',
            feature: 'Capitalization',
            message: `High capital letter ratio (${(capitals * 100).toFixed(0)}%)`,
            detail: 'Excessive capitals can indicate promotional or fake content'
        });
    }
    
    // Average Word Length Analysis
    const avgWordLen = features.avg_word_length || 0;
    if (avgWordLen < 2.5) {
        warnings.push({
            severity: 'low',
            feature: 'Word Length',
            message: `Very short words on average (${avgWordLen.toFixed(1)} characters)`,
            detail: 'May indicate simplistic or rushed writing'
        });
    }
    
    // Subjectivity Analysis
    const subjectivity = features.sentiment_subjectivity || 0;
    if (subjectivity > 0.9) {
        warnings.push({
            severity: 'low',
            feature: 'Subjectivity',
            message: `Highly subjective content (${(subjectivity * 100).toFixed(0)}%)`,
            detail: 'Very opinion-heavy without factual details'
        });
    }
    
    return { reasons, warnings };
}

// Calculate feature contribution to anomaly score
function calculateFeatureContributions(features) {
    const contributions = [];
    const profile = FEATURE_PROFILES.normal;
    
    Object.keys(features).forEach(key => {
        if (profile[key]) {
            const value = features[key];
            const { min, max, optimal } = profile[key];
            
            // Calculate deviation from optimal
            let deviation = 0;
            if (value < min) {
                deviation = ((min - value) / min) * 100;
            } else if (value > max) {
                deviation = ((value - max) / max) * 100;
            } else {
                deviation = Math.abs(value - optimal) / (max - min) * 50;
            }
            
            contributions.push({
                feature: key,
                value: value,
                deviation: deviation,
                inRange: value >= min && value <= max
            });
        }
    });
    
    // Sort by deviation (highest first)
    contributions.sort((a, b) => b.deviation - a.deviation);
    return contributions;
}

// Validation functions
function detectTagalog(text) {
    const tagalogWords = [
        'ang', 'ng', 'sa', 'na', 'ay', 'mga', 'ito', 'iyan', 'iyon', 'siya', 'kami', 'tayo', 'kayo',
        'ako', 'ikaw', 'sila', 'namin', 'ninyo', 'nila', 'ko', 'mo', 'niya', 'amin', 'inyo', 'kanila',
        'at', 'o', 'pero', 'kung', 'kapag', 'dahil', 'kaya', 'kasi', 'dapat', 'pwede', 'puwede',
        'maganda', 'mahal', 'mura', 'mabuti', 'masama', 'salamat', 'po', 'opo', 'hindi', 'oo',
        'kumusta', 'kamusta', 'paalam', 'sige', 'tulong', 'tulungan', 'bili', 'bumili', 'bumibili',
        'gusto', 'gusto ko', 'ayaw', 'mahal ko', 'mahal kita', 'mahal na mahal', 'salamat po',
        'pangit', 'maganda', 'masarap', 'masama', 'mabait', 'masungit', 'mabuti', 'masama'
    ];
    
    const textLower = text.toLowerCase();
    const words = textLower.split(/\s+/);
    const tagalogCount = words.filter(word => tagalogWords.includes(word)).length;
    
    return words.length > 0 && (tagalogCount / words.length) > 0.2;
}

function detectGibberish(text) {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    
    if (words.length === 0) return false;
    
    // Check 1: Repetitive character patterns (like "adadada", "ababab", "aaaa")
    for (const word of words) {
        if (word.length >= 3) {
            // Check for repeating 2-3 character patterns
            for (let patternLen = 2; patternLen <= 3; patternLen++) {
                if (word.length >= patternLen * 2) {
                    const pattern = word.substring(0, patternLen);
                    const repeats = (word.match(new RegExp(pattern, 'g')) || []).length;
                    if (repeats >= 3) return true; // Pattern appears 3+ times
                }
            }
            
            // Check for single character repetition (like "aaaa", "bbbb")
            if (word.length >= 4) {
                const charCounts = {};
                for (const char of word) {
                    charCounts[char] = (charCounts[char] || 0) + 1;
                }
                const maxCount = Math.max(...Object.values(charCounts));
                // If one character makes up >70% of the word, likely gibberish
                if (maxCount / word.length > 0.7) return true;
            }
        }
    }
    
    // Check 2: Very short average word length
    const avgLength = words.reduce((sum, w) => sum + w.length, 0) / words.length;
    if (avgLength < 3.5 && words.length > 2) return true;
    
    // Check 3: Very low lexical diversity (stricter)
    const uniqueWords = new Set(words);
    const diversity = uniqueWords.size / words.length;
    if (diversity < 0.4 && words.length > 2) return true;
    
    // Check 4: High ratio of short words with low diversity
    const shortWords = words.filter(w => w.length <= 4);
    const shortRatio = shortWords.length / words.length;
    if (shortRatio > 0.6 && diversity < 0.5) return true;
    
    // Check 5: Unusual consonant clusters (3+ consonants in a row)
    const consonantClusters = words.filter(w => /[bcdfghjklmnpqrstvwxyz]{3,}/i.test(w));
    if (words.length > 0 && (consonantClusters.length / words.length) > 0.4) return true;
    
    // Check 6: All words are very short (1-2 letters)
    if (words.length >= 3) {
        const veryShort = words.filter(w => w.length <= 2);
        if (veryShort.length / words.length > 0.7) return true;
    }
    
    // Check 7: Long words with unusual consonant patterns (like "kjashduikqweha")
    const longWords = words.filter(w => w.length > 8);
    if (longWords.length > 0) {
        const longWordsWithClusters = longWords.filter(w => /[bcdfghjklmnpqrstvwxyz]{3,}/i.test(w));
        const longWordRatio = longWords.length / words.length;
        if (longWordRatio > 0.3 && longWordsWithClusters.length > 0) return true;
    }
    
    // Check 8: Single long word that looks like random typing (like "kjashduikqweha")
    if (words.length === 1 && words[0].length > 10) {
        const word = words[0];
        // Count vowels
        const vowels = (word.match(/[aeiou]/g) || []).length;
        const vowelRatio = vowels / word.length;
        
        // If very few vowels (< 20%) in a long word, likely gibberish
        if (vowelRatio < 0.2) return true;
        
        // Check for 4+ consecutive consonants
        if (/[bcdfghjklmnpqrstvwxyz]{4,}/i.test(word)) return true;
    }
    
    // Check 9: Low character variety within words (like "adadada" - only 2 unique chars)
    for (const word of words) {
        if (word.length >= 4) {
            const uniqueChars = new Set(word).size;
            const charVariety = uniqueChars / word.length;
            // If word has very few unique characters (< 40% variety), likely repetitive gibberish
            if (charVariety < 0.4) return true;
        }
    }
    
    // Check 10: Words that are too repetitive (same 2-3 chars repeating)
    for (const word of words) {
        if (word.length >= 6) {
            const uniqueChars = new Set(word).size;
            // If word is mostly made of 2-3 unique characters, likely gibberish
            if (uniqueChars <= 3) return true;
        }
    }
    
    // Check 11: Single short word that looks like random keyboard mashing (like "qweasd")
    if (words.length === 1) {
        const word = words[0];
        if (word.length >= 4 && word.length <= 8) {
            // Check if it's all lowercase and looks like random typing
            if (word === word.toLowerCase() && /^[a-z]+$/.test(word)) {
                // Check for keyboard patterns (qwerty, asdf, etc.)
                const keyboardRows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
                let consecutiveChars = 0;
                for (let i = 0; i < word.length - 1; i++) {
                    for (const row of keyboardRows) {
                        const idx1 = row.indexOf(word[i]);
                        const idx2 = row.indexOf(word[i + 1]);
                        if (idx1 !== -1 && idx2 !== -1 && Math.abs(idx1 - idx2) <= 2) {
                            consecutiveChars++;
                            break;
                        }
                    }
                }
                
                // If many consecutive keyboard-adjacent chars, likely gibberish
                if (consecutiveChars >= 2) return true;
                
                // Check vowel distribution - random typing often has odd patterns
                const vowels = (word.match(/[aeiou]/g) || []).length;
                const vowelRatio = vowels / word.length;
                // If very few vowels (< 25%) in a short word, likely gibberish
                if (vowelRatio < 0.25) return true;
            }
        }
    }
    
    return false;
}

function validateInput(text) {
    if (!text || text.trim().length === 0) {
        return { valid: false, message: "Please enter a review before submitting." };
    }
    
    // Check word count limit
    const wordCountValue = countWords(text);
    if (wordCountValue > MAX_WORDS) {
        return { valid: false, message: `Review exceeds the maximum limit of ${MAX_WORDS} words. Please shorten your review.` };
    }
    
    if (detectTagalog(text)) {
        return { valid: false, message: "Invalid data entry" };
    }
    
    if (detectGibberish(text)) {
        return { valid: false, message: "Invalid data entry" };
    }
    
    return { valid: true, message: null };
}

// Show invalid data modal
function showInvalidModal(message) {
    if (invalidMessage) {
        invalidMessage.textContent = message || "Invalid data entry";
    }
    if (invalidModal) {
        invalidModal.classList.add('show');
    }
}

// Close invalid modal
if (invalidOkBtn) {
    invalidOkBtn.addEventListener('click', () => {
        if (invalidModal) {
            invalidModal.classList.remove('show');
        }
    });
}

// Close invalid modal on outside click
if (invalidModal) {
    invalidModal.addEventListener('click', (event) => {
        if (event.target === invalidModal) {
            invalidModal.classList.remove('show');
        }
    });
}

// Handle Submit Button Click - Show Terms First
submitBtn.addEventListener('click', () => {
    const reviewText = reviewInput.value.trim();
    
    // Validate input before showing terms modal
    const validation = validateInput(reviewText);
    if (!validation.valid) {
        // Check if it's a word count issue
        const wordCountValue = countWords(reviewText);
        if (wordCountValue > MAX_WORDS) {
            showInvalidModal("Invalid data entry");
        } else {
            showInvalidModal(validation.message);
        }
        return;
    }

    termsModal.classList.add('show');
});

// Handle Terms Accept
termsAcceptBtn.addEventListener('click', async () => {
    const reviewText = reviewInput.value.trim();
    
    // Validate again before sending to API
    const validation = validateInput(reviewText);
    if (!validation.valid) {
        termsModal.classList.remove('show');
        showInvalidModal(validation.message);
        return;
    }
    
    termsModal.classList.remove('show');
    
    resultTitle.textContent = "Analyzing...";
    resultMessage.textContent = "Please wait while we process your review.";

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ review: reviewText })
        });

        const data = await response.json();
        console.log("Raw backend response:", data);
        
        // Check for validation errors from backend
        if (data.error && data.error.includes("Invalid data entry")) {
            showInvalidModal("Invalid data entry");
            resultTitle.textContent = "Invalid Input";
            resultMessage.textContent = "The input contains invalid data. Please enter a valid English review.";
            return;
        }

        let result;
        if (data.result) {
            result = data.result;
        } else {
            result = data;
        }

        const isGenuine = result.prediction.toLowerCase() === "normal";
        
        if (isGenuine) {
            resultTitle.textContent = "✓ Genuine Review";
        } else {
            resultTitle.textContent = "⚠ Suspicious Review";
        }

        const distance = Number(result.distance || 0);
        const threshold = Number(result.threshold || 0);
        const margin = distance - threshold;
        const confidence = Number(result.confidence || 0);
        const features = result.features || {};

        // Analyze features with detailed reasoning
        const analysis = analyzeFeatures(features, result.prediction);
        const contributions = calculateFeatureContributions(features);
        
        // Build reasons HTML
        let reasonsHtml = '';
        if (analysis.reasons.length > 0) {
            reasonsHtml = '<div style="margin: 8px 0; font-size: 0.9rem;">';
            reasonsHtml += (
                '<p style="margin: 4px 0 6px; font-weight: 600; ' +
                'color: var(--secondary-color); font-size: 0.95rem;">' +
                'Key Warning Signs:</p>'
            );
            analysis.reasons.forEach(reason => {
                const severityColor = (
                    reason.severity === 'high' ? '#d32f2f' :
                    reason.severity === 'medium' ? '#f57c00' : '#fbc02d'
                );
                reasonsHtml += `
                    <div style=\"margin: 6px 0; padding: 8px; background: #fff3e0; border-left: 4px solid ${severityColor}; border-radius: 4px;\">
                        <div style=\"font-weight: 600; color: ${severityColor}; margin-bottom: 2px; font-size: 0.92rem;\">
                            ${reason.feature}: ${reason.message}
                        </div>
                        <div style=\"font-size: 0.85rem; color: #666;\">
                            ${reason.detail}
                        </div>
                    </div>
                `;
            });
            reasonsHtml += '</div>';
        } else if (isGenuine) {
            reasonsHtml += (
                '<div style="margin: 10px 0; padding: 12px; ' +
                'background: #e8f5e9; border-left: 4px solid #4caf50; ' +
                'border-radius: 4px;">'
            );
            reasonsHtml += (
                '<p style="margin: 0; color: #2e7d32; font-weight: 500;">' +
                'All features are within normal ranges. This review shows ' +
                'characteristics of genuine feedback.</p>'
            );
            reasonsHtml += '</div>';
        }

        // Add minor warnings if any
        if (analysis.warnings.length > 0 && !isGenuine) {
            reasonsHtml += (
                '<p style="margin: 12px 0 6px; font-weight: 600; ' +
                'color: #666; font-size: 0.9rem;">Minor Observations:</p>'
            );
            analysis.warnings.forEach(warning => {
                reasonsHtml += `
                    <div style="margin: 6px 0; padding: 8px; background: #f5f5f5; border-left: 3px solid #999; border-radius: 4px; font-size: 0.9rem;">
                        <strong>${warning.feature}:</strong> ${warning.message}
                    </div>
                `;
            });
        }

        // Build feature contributions table
        const topContributions = contributions.slice(0, 5);
        const contributionsHtml = topContributions.map(c => {
            const formatted = typeof c.value === 'number'
                ? (Math.abs(c.value) < 1 && c.value !== 0
                    ? c.value.toFixed(4)
                    : c.value.toFixed(2))
                : String(c.value);
            const statusClass = c.inRange
                ? 'status-tag status-pass'
                : 'status-tag status-warn';
            const statusText = c.inRange ? 'Normal' : 'Unusual';
            const deviationBar = Math.min(c.deviation, 100);
            const barColor = c.inRange ? '#4caf50' : '#f57c00';
            const niceName = FEATURE_LABELS[c.feature] ||
                c.feature.replace(/_/g, ' ')
                    .replace(/\b\w/g, l => l.toUpperCase());
            const desc = FEATURE_DESCRIPTIONS[c.feature] || '';
            
            return `
                <div style="margin: 8px 0; padding: 10px; background: #f9f9f9; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600;">${niceName}
                            <span class="help-icon" title="${desc}">?</span>
                        </span>
                        <span class="${statusClass}">${statusText}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="min-width: 60px; font-size: 0.95rem;">Value: <strong>${formatted}</strong></span>
                        <div style="flex: 1; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                            <div style="height: 100%; width: ${deviationBar}%; background: ${barColor}; transition: width 0.3s;"></div>
                        </div>
                        <span style="min-width: 50px; font-size: 0.9rem; color: #666;">${deviationBar.toFixed(0)}%</span>
                    </div>
                </div>
            `;
        }).join('');

        const badgeClass = isGenuine ? 'badge badge-normal' : 'badge badge-anom';
        const barClass = isGenuine ? 'progress-bar progress-blue' : 'progress-bar progress-red';
        const safeConfidence = Math.max(0, Math.min(100, confidence));

        resultMessage.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span class="${badgeClass}">${isGenuine ? 'Normal' : 'Suspicious'}</span>
            </div>

            ${reasonsHtml}

            <div class="details-toggle" id="details-toggle">Show all technical details</div>
            <div class="details-content" id="details-content">
                <p style="margin: 16px 0 8px; font-weight: 600; color: var(--secondary-color);">All Feature Values</p>
                <div style="overflow-x:auto;">
                    <table style="width:100%; border-collapse: collapse; font-size: 0.9rem; background: white;">
                        <thead>
                            <tr style="background: #f5f5f5;">
                                <th style="text-align:left; padding:10px 8px; border-bottom: 2px solid #ddd;">Feature</th>
                                <th style="text-align:center; padding:10px 8px; border-bottom: 2px solid #ddd;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(features).map(([key, value]) => {
                                const formatted = typeof value === 'number'
                                    ? (Math.abs(value) < 1 && value !== 0
                                        ? value.toFixed(4)
                                        : value.toFixed(3))
                                    : String(value);
                                const profile = FEATURE_PROFILES.normal[key];
                                const inRange = profile
                                    ? (value >= profile.min && value <= profile.max)
                                    : true;
                                const statusClass = inRange
                                    ? 'status-tag status-pass'
                                    : 'status-tag status-warn';
                                const statusText = inRange ? 'Normal' : 'Unusual';
                                const niceName = FEATURE_LABELS[key] ||
                                    key.replace(/_/g, ' ')
                                        .replace(/\b\w/g, l => l.toUpperCase());
                                const desc = FEATURE_DESCRIPTIONS[key] || '';
                                const basis = profile
                                    ? `Typical: ${Number(profile.min).toFixed(3)}–` +
                                      `${Number(profile.max).toFixed(3)} ` +
                                      `(optimal ${Number(profile.optimal).toFixed(3)})`
                                    : '';
                                
                                return `
                                    <tr style="border-bottom: 1px solid #eee;">
                                        <td style="padding:10px 8px;">${niceName}
                                            <span class="help-icon" title="${desc}">?</span>
                                        </td>
                                        <td style="padding:10px 8px; text-align:center;">
                                            <span class="${statusClass}" title="Value: ${formatted}${basis ? ` | ${basis}` : ''}">${statusText}</span>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 10px; text-align: right;">
                    <button id="glossary-btn" class="btn btn-secondary" style="padding: 10px 16px; font-size: 0.95rem;">Feature Glossary</button>
                </div>
            </div>
        `;

        // Toggle handler
        const toggleEl = document.getElementById('details-toggle');
        const detailsEl = document.getElementById('details-content');
        if (toggleEl && detailsEl) {
            // Keep hidden by default; reveal only on click
            detailsEl.classList.remove('show');
            toggleEl.textContent = 'Show all technical details';
            toggleEl.addEventListener('click', () => {
                const showing = detailsEl.classList.toggle('show');
                toggleEl.textContent = showing ? 'Hide technical details' : 'Show all technical details';
            });
        }
        // Sync heights after content update
        syncResultPanelHeight();

        // Glossary button: redirect to Feature Criteria panel and highlight button
        const glossaryBtn = document.getElementById('glossary-btn');
        if (glossaryBtn) {
            glossaryBtn.addEventListener('click', () => {
                const learnMoreBtn = document.getElementById('feature-learn-more');
                const criteriaPanel = document.querySelector(
                    '.feature-criteria-content'
                );
                const glossaryEl = document.getElementById('feature-glossary');
                const target = learnMoreBtn || criteriaPanel || glossaryEl;

                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                }

                // Highlight the Learn More button to draw attention
                if (learnMoreBtn) {
                    // Keep the page from auto-scrolling due to focus
                    if (learnMoreBtn.focus) {
                        learnMoreBtn.focus({ preventScroll: true });
                    }

                    const originalTransition = learnMoreBtn.style.transition;
                    const originalBoxShadow = learnMoreBtn.style.boxShadow;
                    const originalTransform = learnMoreBtn.style.transform;

                    learnMoreBtn.style.transition = (
                        'box-shadow 0.3s, transform 0.2s'
                    );

                    let pulses = 0;
                    const pulse = () => {
                        const shadowValue = (
                            '0 0 0 4px rgba(193, 18, 31, 0.25), ' +
                            '0 0 12px rgba(193, 18, 31, 0.6)'
                        );
                        learnMoreBtn.style.boxShadow = shadowValue;
                        learnMoreBtn.style.transform = 'scale(1.04)';
                        setTimeout(() => {
                            learnMoreBtn.style.boxShadow = (
                                '0 0 0 0 rgba(0,0,0,0)'
                            );
                            learnMoreBtn.style.transform = 'scale(1.0)';
                            pulses++;
                            if (pulses < 3) {
                                setTimeout(pulse, 200);
                            } else {
                                setTimeout(() => {
                                    learnMoreBtn.style.transition = (
                                        originalTransition
                                    );
                                    learnMoreBtn.style.boxShadow = (
                                        originalBoxShadow
                                    );
                                    learnMoreBtn.style.transform = (
                                        originalTransform
                                    );
                                }, 600);
                            }
                        }, 450);
                    };
                    pulse();
                }
            });
        }

        // Match heights: set result panel height equal to left tool panel height
        const leftPanel = document.querySelector('.tool-box.left');
        const rightPanel = document.querySelector('.sticky-results .result-panel');
        if (leftPanel && rightPanel) {
            const leftHeight = leftPanel.getBoundingClientRect().height;
            rightPanel.style.maxHeight = `${leftHeight - 54}px`;
            rightPanel.style.overflowY = 'auto';
        }
    } catch (error) {
        console.error("Error:", error);
        resultTitle.textContent = "Error!";
        resultMessage.textContent = (
            "There was an issue connecting to the server. " +
            "Please try again later."
        );
    }
});

// Handle Terms Decline
termsDeclineBtn.addEventListener('click', () => {
    termsModal.classList.remove('show');
});

// Close modal on outside click
termsModal.addEventListener('click', (event) => {
    if (event.target === termsModal) {
        termsModal.classList.remove('show');
    }
});

// Feature Glossary Toggle
document.addEventListener('DOMContentLoaded', () => {
    const featureLearnMoreBtn = document.getElementById('feature-learn-more');
    const featureGlossary = document.getElementById('feature-glossary');

    if (featureLearnMoreBtn && featureGlossary) {
        // Set initial label to match current visibility
        const isHiddenInit = (
            getComputedStyle(featureGlossary).display === 'none'
        );
        featureLearnMoreBtn.textContent = isHiddenInit
            ? 'Learn More About Features'
            : 'Hide Features';

        featureLearnMoreBtn.addEventListener('click', () => {
            const isHidden = (
                getComputedStyle(featureGlossary).display === 'none'
            );
            featureGlossary.style.display = isHidden ? 'block' : 'none';
            featureLearnMoreBtn.textContent = isHidden
                ? 'Hide Features'
                : 'Learn More About Features';

            // Scroll to glossary if showing
            if (isHidden) {
                setTimeout(() => {
                    featureGlossary.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                }, 100);
            }
        });
    }
});


// Smooth scrolling for navigation links with custom duration
document.addEventListener('DOMContentLoaded', () => {
    const nav = document.querySelector('nav');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

    // Smooth scroll function with custom easing
    function smoothScrollTo(targetPosition, duration = 1000) {
        const startPosition = window.pageYOffset;
        const distance = targetPosition - startPosition;
        let startTime = null;

        function animation(currentTime) {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const progress = Math.min(timeElapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const ease = progress < 0.5 
                ? 4 * progress * progress * progress 
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;
            
            window.scrollTo(0, startPosition + distance * ease);

            if (timeElapsed < duration) {
                requestAnimationFrame(animation);
            }
        }

        requestAnimationFrame(animation);
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const navHeight = nav ? nav.offsetHeight : 0;
                const targetPosition = targetSection.offsetTop - navHeight - 20;
                
                // Use custom smooth scroll with 800ms duration
                smoothScrollTo(targetPosition, 1200);
                
                // Update URL without jumping
                history.pushState(null, '', `#${targetId}`);
            }
        });
    });
});