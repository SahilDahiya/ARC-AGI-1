## Engineering Direction

- No fallback behavior, ever.
- Fail hard on failure conditions.
- No backward compatibility guarantees.
- Forward-only development: do not carry legacy baggage.
- Default to a strict TDD workflow for durable feature development; exploratory research spikes may be exempt until promoted into maintained code.
- Use full red-to-green cycles: write a failing test first, implement the minimal change to pass, then refactor safely.
