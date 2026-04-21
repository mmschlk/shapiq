Fix GitHub issue $ARGUMENTS.

Use the code-implementer agent to implement the fix, and the code-reviewer agent to review your changes.

Overall, follow these steps:
1. Run `gh issue view $ARGUMENTS` to read the issue
2. Find relevant files in shapiq//c
3. Write a failing test that reproduces the issue/bug
4. Fix the issue
5. Verify tests pass with pytest
