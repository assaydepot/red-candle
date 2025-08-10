# AWS S3 Permissions Setup

## Overview

This directory contains AWS configuration files for setting up secure S3 storage for HuggingFace model caching.

## Permission Strategy

### 🔒 Bucket-Level Security

1. **HTTPS Only**: All connections must use TLS/SSL
2. **No Public Access**: Bucket is completely private
3. **Versioning Enabled**: Protects against accidental deletions
4. **Block Public ACLs**: Prevents accidental exposure

### 👤 IAM Users & Permissions

#### CI User (Read-Only)
- **Purpose**: GitHub Actions CI/CD
- **Permissions**: 
  - List bucket contents
  - Download objects
  - Cannot upload, modify, or delete
- **Use in**: GitHub Actions workflow

#### Admin User (Read-Write)
- **Purpose**: Uploading/updating cache
- **Permissions**:
  - Full read access
  - Upload new objects
  - Delete objects
  - Modify bucket settings
- **Use by**: Developers maintaining cache

## Files

- `s3-bucket-policy.json` - Bucket policy (enforces HTTPS)
- `iam-policy-ci-readonly.json` - CI user permissions (minimal)
- `iam-policy-admin.json` - Admin user permissions (full access)
- `setup-aws.sh` - Automated setup script

## Quick Setup

```bash
# Run the automated setup
./aws/setup-aws.sh

# This will:
# 1. Create S3 bucket with security settings
# 2. Create IAM users
# 3. Generate access keys
# 4. Apply all policies
```

## Manual Setup

### 1. Create Bucket
```bash
aws s3 mb s3://red-candle-hf-cache --region us-east-1
```

### 2. Apply Bucket Policy
```bash
aws s3api put-bucket-policy \
  --bucket red-candle-hf-cache \
  --policy file://aws/s3-bucket-policy.json
```

### 3. Block Public Access
```bash
aws s3api put-public-access-block \
  --bucket red-candle-hf-cache \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### 4. Create CI User
```bash
# Create user
aws iam create-user --user-name red-candle-ci-readonly

# Create policy
aws iam create-policy \
  --policy-name red-candle-s3-readonly \
  --policy-document file://aws/iam-policy-ci-readonly.json

# Attach policy
aws iam attach-user-policy \
  --user-name red-candle-ci-readonly \
  --policy-arn arn:aws:iam::YOUR_ACCOUNT:policy/red-candle-s3-readonly

# Create access key
aws iam create-access-key --user-name red-candle-ci-readonly
```

## Security Best Practices

✅ **DO**:
- Use IAM users with minimal permissions
- Rotate access keys every 90 days
- Enable MFA for admin users
- Use versioning to protect against accidental deletion
- Monitor access logs

❌ **DON'T**:
- Make bucket public
- Use root account credentials
- Share access keys in code
- Grant unnecessary permissions
- Disable HTTPS requirement

## Cost Optimization

- **Lifecycle Rules**: Delete old versions after 30 days
- **Storage Class**: Use S3 Standard for frequently accessed cache
- **Region**: Same as GitHub Actions runners (usually us-east-1)

## Monitoring

Check bucket size and costs:
```bash
# Size
aws s3 ls --summarize --recursive s3://red-candle-hf-cache/

# Costs (requires Cost Explorer API)
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --filter file://cost-filter.json
```

## Troubleshooting

### Access Denied
- Check IAM policy is attached to user
- Verify bucket name is correct
- Ensure HTTPS is being used

### Slow Downloads
- Check if bucket and CI are in same region
- Consider using S3 Transfer Acceleration

### High Costs
- Review lifecycle policies
- Check for unnecessary versions
- Monitor request patterns