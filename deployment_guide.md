# Deploying Hey Attrangi Bot to Hugging Face Spaces

This guide will help you deploy your Attrangi Bot to Hugging Face Spaces.

## Prerequisites
- A [Hugging Face](https://huggingface.co/) account.
- Your project code pushed to a GitHub repository (optional but recommended) or ready to upload.

## Step 1: Create a New Space
1.  Log in to Hugging Face.
2.  Go to **New Space**.
3.  Enter a name (e.g., `hey-attrangi-bot`).
4.  Select **Docker** as the SDK.
5.  Choose **Blank** for the template.
6.  Click **Create Space**.

## Step 2: Set Up Environment Variables (Secrets)
**CRITICAL**: Do NOT upload your `.env` file. You must set these as secrets in the Space settings.

1.  Go to your Space's **Settings** tab.
2.  Scroll down to **Variables and secrets**.
3.  Click **New secret** and add the following:
    *   `GROQ_API_KEY`: *(Paste your Groq API key)*
    *   `DB_CONNECTION_STRING`: *(Paste your Neon DB connection string)*

## Step 3: Upload Code
You can upload your code via git.

### Method A: Via Command Line (Recommended)
1.  Clone your Space's repository (you'll see the command in the "App" tab if it's empty):
    ```bash
    git clone https://huggingface.co/spaces/YOUR_USERNAME/hey-attrangi-bot
    ```
2.  Copy all your project files into this new folder **EXCEPT** `.env`, `vector_store`, and `data`.
    *   *Tip: The provided `.gitignore` and `.dockerignore` will help with this if you just copy everything.*
3.  Push the changes:
    ```bash
    cd hey-attrangi-bot
    git add .
    git commit -m "Initial deployment"
    git push
    ```

### Method B: Via Web Upload
1.  Go to the **Files** tab of your Space.
2.  Click **Add file** -> **Upload files**.
3.  Drag and drop your project files (`main.py`, `Dockerfile`, `requirements.txt`, etc.).
4.  Commit the changes.

## Step 4: Watch it Build
1.  Go to the **App** tab.
2.  You will see a "Building" status.
3.  Click **Logs** to see the progress.
4.  Once built, your bot will be live!

## Notes
- **Database**: Since we migrated to Neon Postgres, your chat summaries will successfully save to the cloud database even when the Space restarts.
- **Memory**: The active chat history (live conversation) is ephemeral and will reset if the Space restarts (which happens occasionally). This is expected behavior for this architecture.
- **Knowledge Base**: The app rebuilds the vector store on startup from PDFs in `knowledge_base` or `useable_pdfs`. Ensure your PDFs are included in the upload/push.
