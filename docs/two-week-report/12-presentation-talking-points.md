# Presentation Talking Points

This is a 5–7 minute, snapshot-based script about the project as it exists in the available workspace.
It deliberately does not claim that the described features were your work during the last two weeks; confirm and replace the scope caveat with verified Git evidence before presenting.

## Opening — about 30 seconds

“I’ll give a short update on VERIMIND, a prototype that checks factual claims in generated text. One important note: the workspace I reviewed has the current project files but not its Git history, so I can describe what the project currently does, but I can’t responsibly attribute specific changes to the last two weeks yet.”

## Top achievements visible in the project — about 2 minutes

“The core flow takes a block of generated text and breaks it into individual factual claims. That makes it possible to evaluate the statements separately instead of giving one vague score for an entire response.”

“For each claim, the system looks for supporting evidence, estimates whether that evidence supports or contradicts the claim, and returns a risk score and a plain-language explanation. It keeps the evidence alongside the result so a reviewer can inspect why a claim was flagged.”

“The project can be used through an API and a small dashboard. The dashboard shows the overall risk and then the verdict, explanation, and evidence for each claim.”

“It is designed to work locally, with SQLite persistence and local retrieval. Optional web search can add external evidence when configured, while the README describes local-only retrieval as the fallback.”

“The repository snapshot also includes tests and an automated CI workflow for linting, type checking, and running the test suite. I have not run those checks as part of preparing this report.”

## Challenges — about 1 minute

“The main product challenge visible in the code is balancing useful evidence retrieval with a lightweight, locally runnable prototype. The README explicitly notes that the current approach uses TF-IDF and a lightweight classifier rather than a larger NLI model.”

“I don’t want to imply that this was a specific challenge I solved in the last two weeks without the commit history. The same is true for bug fixes or design decisions: those need to be matched to actual commits and confirmed where the reason is not documented.”

## Impact and numbers — about 45 seconds

“The observable impact today is an end-to-end flow from text input to claim-level verdicts, evidence, and a saved report. The test notebook covers several of those core components.”

“I don’t have reliable two-week numbers for commits, files changed, lines added or removed, or measured user and performance impact. Those numbers are unavailable because the workspace does not include Git history, and the current metrics endpoint returns placeholder values.”

## Next steps — about 45 seconds

“My first step is to bring in the Git-enabled checkout or the 14-day log and diff, so I can give an accurate timeline and separate what changed from what was already present.”

“After that, I’ll confirm the motivation and impact for each change, then update the summary and presentation. As a product follow-up, the visible limitations and placeholder metrics can be reviewed against the team's requirements.”

## Closing — about 20 seconds

“In summary, VERIMIND currently demonstrates an explainable claim-verification flow with local evidence, optional web search, an API, a dashboard, and persistence. I’ll make the retrospective specific to my last two weeks once the missing history and change context are available.”

## Likely manager questions

1. **What did you personally complete in the last two weeks?**  
   “This workspace snapshot does not contain the commit history needed to answer accurately. I’ll map the changes to my commits before giving a definitive list.”

2. **How many commits or files did you change?**  
   “I can’t verify those numbers from this checkout because Git metadata is missing. I don’t want to guess.”

3. **How does VERIMIND work?**  
   “It splits text into claims, retrieves evidence, evaluates support or contradiction, and returns a verdict, risk score, explanation, and evidence for each claim.”

4. **What does a user see?**  
   “A user can submit text through the dashboard or API and see an overall risk measure plus claim-by-claim results and evidence.”

5. **What is the biggest current limitation?**  
   “The README describes the model and retrieval approach as lightweight and prototype-oriented. It also notes that live web search is optional.”

6. **How do we know it works?**  
   “The project includes tests for segmentation, retrieval of known facts, model behavior, scoring, pipeline output, and API behavior. I have not run them for this report, so I can’t give a fresh pass rate.”

7. **What is the measured business or performance impact?**  
   “No reliable measurement is available in this snapshot. In particular, the current metrics endpoint returns placeholder values, so I would first define and collect the metrics we need.”

8. **Were there database or API changes?**  
   “The current project includes SQLite report storage and verification/report API routes. I can’t confirm whether their schemas or behavior changed during these two weeks without the prior revision.”

9. **What will you do next?**  
   “First, verify the change list and reasons from Git history. Then I can report accurate counts and decide with the team which visible limitations deserve priority.”

10. **Why can’t this report give a detailed timeline?**  
    “The provided folder is a source snapshot without `.git` metadata, and Git is not installed in this environment. A Git-enabled checkout or exported log and diffs would resolve that.”

See [00-overview-summary.md](./00-overview-summary.md) for scope and counts, and [11-pending-and-next-steps.md](./11-pending-and-next-steps.md) for suggested follow-up.
