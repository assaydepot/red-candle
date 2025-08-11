# GitHub Secrets Setup Checklist

## ✅ Steps to Complete

### 1. Add AWS Credentials to GitHub Secrets

Go to your repository's secrets page:
```
https://github.com/YOUR_USERNAME/red-candle/settings/secrets/actions
```

Or navigate manually:
1. Go to your repository on GitHub
2. Click "Settings" tab
3. In left sidebar: Security → Secrets and variables → Actions
4. Click "New repository secret"

### 2. Add These Secrets

Add two secrets with the EXACT names:

1. **AWS_ACCESS_KEY_ID**
   - Name: `AWS_ACCESS_KEY_ID`
   - Value: The access key ID from the setup script (starts with `AKIA...`)

2. **AWS_SECRET_ACCESS_KEY**
   - Name: `AWS_SECRET_ACCESS_KEY`  
   - Value: The secret access key from the setup script (longer random string)

⚠️ **IMPORTANT**: Copy these values exactly, including any special characters!

### 3. Verify Setup

After adding the secrets, you should see:
- AWS_ACCESS_KEY_ID (Repository secret)
- AWS_SECRET_ACCESS_KEY (Repository secret)

### 4. Test the Workflow

Create a test PR to trigger the workflow:
```bash
git checkout -b test-s3-cache
echo "# Testing S3 cache" >> README.md
git add README.md
git commit -m "Test S3 cache in CI"
git push origin test-s3-cache
```

Then open a PR on GitHub.

## 📊 What to Expect in CI

When the workflow runs, you should see:

1. **AWS Configuration**: ✅ Success (uses your secrets)
2. **S3 Sync**: Downloads cache from S3
3. **Cache Info**: Shows size and model count
4. **Tests Run**: With `HF_HUB_OFFLINE=1` (no API calls)
5. **No 429 Errors**: No rate limiting!

## 🔍 Troubleshooting

### "Invalid AWS credentials"
- Double-check the secret names (must be exact)
- Verify you copied the full key values
- Check if keys have expired (regenerate if needed)

### "Access Denied to S3"
- Verify bucket name is `red-candle-hf-cache`
- Check IAM policy is attached to user
- Confirm region is `eu-central-1`

### "Models still downloading"
- Check if HF_HUB_OFFLINE is set
- Verify all needed models are in S3
- Look for model names in error messages

## 🎉 Success Indicators

You'll know it's working when:
- No HuggingFace 429 errors
- Tests run faster (no downloads)
- CI logs show "Offline mode enabled"
- All tests pass without network calls to HuggingFace

## 📝 Notes

- The credentials are for a read-only user
- They can only access your specific S3 bucket
- No other AWS services can be accessed
- Keys should be rotated every 90 days

## 🔐 Security

Your secrets are:
- Encrypted at rest by GitHub
- Only visible to workflows in your repo
- Not exposed to PR forks
- Not logged in workflow output