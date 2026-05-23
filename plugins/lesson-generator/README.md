# Lesson Generator Plugin

Generate compact, standalone, multi-lesson course artifacts — complete with lesson navigation, learning objectives, flashcards, quizzes, and source links — as a single browser-ready HTML/CSS/JS bundle.

## Features

- **Multi-Lesson Courses**: 6-8 ordered lessons by default, each with its own page
- **Lesson Navigation**: Sidebar / table of contents with active lesson reader
- **Learning Objectives**: 2-4 focused objectives per lesson
- **Flashcards**: 2-3 flip-in-place flashcards per lesson
- **Knowledge Checks**: 1-2 quiz questions per lesson with immediate feedback
- **Cumulative Review**: Final quiz that synthesizes the full topic
- **Source Cards**: Real clickable links when source material is provided
- **Self-Contained**: Plain HTML/CSS/JS — no backend, no database, no external services
- **Warm Learning UI**: Polished design tokens tuned for long-form reading

## Usage

Invoke this skill when you want to:
- "Build me a course on..."
- "Create an interactive lesson about..."
- "Make a study guide for..."
- "Generate flashcards and quizzes for..."
- "Turn this topic into a mini-course"

## Examples

### Topic Course
```
Build a 6-lesson course on the basics of reinforcement learning
```

### Study Guide
```
Create an interactive study guide for the SAT vocabulary list I just pasted
```

### Source-Backed Lessons
```
Make a 7-lesson course on prompt engineering with source links from the papers I shared
```

## Output

The skill writes three files to the workspace root:
- `index.html` — full course shell with sidebar, lesson reader, flashcards, quiz, review
- `styles.css` — warm reading-optimized design tokens
- `script.js` — course data array, lesson navigation, flashcard flip, quiz feedback

Open `index.html` in a browser to use the course.

## Version

- Version: 1.0.0
- Author: DAIR.AI
