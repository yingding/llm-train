## Memory System

This project uses aidiary for persistent memory. The MCP server
"copilot-memory" provides tools for reading and writing structured knowledge.

### Session start
Call `briefing` as the first action in every conversation. This returns:
- Top conventions (what to follow)
- Recent anti-patterns (what to avoid)
- Health warnings (stale entries, contradictions)

### During the session
- Before acting on a topic, call `recall` with relevant keywords
- After learning something new, call `remember` with file, heading, body, confidence, source
- After making a mistake, call `record_mistake` with what happened, why it was wrong, and the lesson
- Use `stage` for entries that need human review before committing

### Session end
Call `reflect` to summarize what was learned, verified, and corrected.
