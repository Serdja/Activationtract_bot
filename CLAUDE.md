# Activationtract Bot

## MemPalace Integration

MemPalace is installed and configured for this project. It stores conversation history and enables semantic search across past sessions.

### Usage

```bash
# Search past conversations
mempalace search "your query"

# Load context at session start
mempalace wake-up

# Mine new project files after changes
mempalace mine /home/user/Activationtract_bot

# Mine Claude Code sessions
mempalace mine ~/.claude/projects/ --mode convos

# Check status
mempalace status
```
