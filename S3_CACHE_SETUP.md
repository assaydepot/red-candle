# S3 Cache Setup for HuggingFace Models

## Overview

This approach uses an S3 bucket to store a known-good set of HuggingFace models, avoiding rate limiting and ensuring consistent test environments.

## Benefits

- **No rate limiting**: Downloads from S3, not HuggingFace
- **Full control**: You manage what models are cached
- **Fast in CI**: AWS to AWS transfers are very fast
- **Offline mode**: With complete cache, tests run without any API calls
- **Cost effective**: S3 storage is cheap (~$0.023/GB/month)

## Setup Instructions

### 1. Prepare Your Local Cache

```bash
# Option A: Use your existing cache
# Your existing ~/.cache/huggingface already has the models you need

# Option B: Backup and rebuild
cp -r ~/.cache/huggingface ~/.cache/huggingface.backup
bundle exec rspec  # Run specs to populate cache with needed models
```

### 2. Upload to S3

```bash
chmod +x script/setup_s3_cache.sh
./script/setup_s3_cache.sh
# Follow the prompts
```

Or manually:

```bash
# Create bucket
aws s3 mb s3://your-bucket-name --region us-east-1

# Upload cache
aws s3 sync ~/.cache/huggingface s3://your-bucket-name/cache/ \
  --exclude ".DS_Store" \
  --exclude "*.pyc"

# Verify
aws s3 ls --summarize --recursive s3://your-bucket-name/cache/
```

### 3. Configure GitHub Actions

1. Add AWS credentials to GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Uncomment the S3 sections in `.github/workflows/build.yml`

3. Update the bucket name in the workflow

### 4. IAM Permissions

Create an IAM user with minimal permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

## Cost Estimation

- **Storage**: ~1-5GB of models = $0.02-0.12/month
- **Transfer**: Free within same AWS region
- **Requests**: Negligible for CI runs

## Troubleshooting

### Models still downloading?

1. Check cache completeness:
```bash
ls -la ~/.cache/huggingface/hub/models--*
```

2. Ensure HF_HUB_OFFLINE is set:
```bash
export HF_HUB_OFFLINE=1
bundle exec rake test
```

### S3 sync slow?

Use `--only-show-errors` flag:
```bash
aws s3 sync s3://bucket/cache/ ~/.cache/huggingface/ --only-show-errors
```

### Want to minimize cache size?

Clear cache and run specs to get only what's needed:
```bash
rm -rf ~/.cache/huggingface
bundle exec rspec
# This downloads only the models your specs actually use
```

## Alternative: Public S3 Bucket

If you want to share the cache publicly (for other contributors):

1. Make bucket public:
```bash
aws s3api put-bucket-acl --bucket your-bucket-name --acl public-read
```

2. Use wget/curl in CI instead of AWS CLI:
```bash
wget -r -np -nH --cut-dirs=1 https://your-bucket-name.s3.amazonaws.com/cache/
```

## Maintenance

- **Update cache**: Re-run setup script when adding new models
- **Monitor size**: Check S3 bucket size periodically
- **Rotate credentials**: Change AWS keys every 90 days
- **Clean up**: Remove unused models to keep cache minimal