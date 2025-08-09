#!/bin/bash
# Script to set up AWS S3 bucket and IAM users for Red Candle CI

set -e

BUCKET_NAME="red-candle-hf-cache"
AWS_REGION="eu-central-1"

echo "=== AWS Setup for Red Candle HuggingFace Cache ==="
echo

# Step 1: Create S3 bucket
echo "1. Creating S3 bucket..."
if aws s3 ls "s3://$BUCKET_NAME" 2>/dev/null; then
    echo "   Bucket already exists"
else
    aws s3 mb "s3://$BUCKET_NAME" --region $AWS_REGION
    echo "   Bucket created: $BUCKET_NAME"
fi

# Step 2: Apply bucket policy (require HTTPS)
echo
echo "2. Applying bucket policy..."
aws s3api put-bucket-policy \
    --bucket $BUCKET_NAME \
    --policy file://aws/s3-bucket-policy.json
echo "   Policy applied: HTTPS required"

# Step 3: Enable versioning (optional but recommended)
echo
echo "3. Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled
echo "   Versioning enabled"

# Step 4: Block public access (security best practice)
echo
echo "4. Blocking public access..."
aws s3api put-public-access-block \
    --bucket $BUCKET_NAME \
    --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
echo "   Public access blocked"

# Step 5: Create IAM user for CI (read-only)
echo
echo "5. Creating IAM user for CI..."
CI_USER="red-candle-ci-readonly"

if aws iam get-user --user-name $CI_USER 2>/dev/null; then
    echo "   User already exists: $CI_USER"
else
    aws iam create-user --user-name $CI_USER
    echo "   User created: $CI_USER"
fi

# Step 6: Create and attach policy
echo
echo "6. Creating IAM policy for CI..."
POLICY_NAME="red-candle-s3-readonly"

# Check if policy exists
POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='$POLICY_NAME'].Arn" --output text)

if [ -z "$POLICY_ARN" ]; then
    POLICY_ARN=$(aws iam create-policy \
        --policy-name $POLICY_NAME \
        --policy-document file://aws/iam-policy-ci-readonly.json \
        --query 'Policy.Arn' \
        --output text)
    echo "   Policy created: $POLICY_NAME"
else
    echo "   Policy already exists: $POLICY_NAME"
    # Update the policy version
    aws iam create-policy-version \
        --policy-arn $POLICY_ARN \
        --policy-document file://aws/iam-policy-ci-readonly.json \
        --set-as-default 2>/dev/null || true
fi

# Attach policy to user
aws iam attach-user-policy \
    --user-name $CI_USER \
    --policy-arn $POLICY_ARN
echo "   Policy attached to user"

# Step 7: Create access key for CI user
echo
echo "7. Creating access key for CI..."
echo
echo "   ⚠️  IMPORTANT: Save these credentials immediately!"
echo "   They will only be shown once."
echo

read -p "Create new access key? [y/N]: " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    KEY_OUTPUT=$(aws iam create-access-key --user-name $CI_USER)

    ACCESS_KEY_ID=$(echo $KEY_OUTPUT | jq -r '.AccessKey.AccessKeyId')
    SECRET_ACCESS_KEY=$(echo $KEY_OUTPUT | jq -r '.AccessKey.SecretAccessKey')

    echo
    echo "   ===== SAVE THESE CREDENTIALS ====="
    echo "   AWS_ACCESS_KEY_ID: $ACCESS_KEY_ID"
    echo "   AWS_SECRET_ACCESS_KEY: $SECRET_ACCESS_KEY"
    echo "   ==================================="
    echo
    echo "   Add these to GitHub Secrets:"
    echo "   1. Go to: https://github.com/YOUR_ORG/red-candle/settings/secrets/actions"
    echo "   2. Add AWS_ACCESS_KEY_ID"
    echo "   3. Add AWS_SECRET_ACCESS_KEY"
fi

# Step 8: Create admin user (for uploads)
echo
echo "8. Creating admin user for uploads..."
ADMIN_USER="red-candle-admin"

read -p "Create admin user for uploads? [y/N]: " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    if aws iam get-user --user-name $ADMIN_USER 2>/dev/null; then
        echo "   User already exists: $ADMIN_USER"
    else
        aws iam create-user --user-name $ADMIN_USER
        echo "   User created: $ADMIN_USER"
    fi

    # Create admin policy
    ADMIN_POLICY_NAME="red-candle-s3-admin"
    ADMIN_POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='$ADMIN_POLICY_NAME'].Arn" --output text)

    if [ -z "$ADMIN_POLICY_ARN" ]; then
        ADMIN_POLICY_ARN=$(aws iam create-policy \
            --policy-name $ADMIN_POLICY_NAME \
            --policy-document file://aws/iam-policy-admin.json \
            --query 'Policy.Arn' \
            --output text)
        echo "   Admin policy created"
    fi

    aws iam attach-user-policy \
        --user-name $ADMIN_USER \
        --policy-arn $ADMIN_POLICY_ARN
    echo "   Admin policy attached"
fi

# Step 9: Summary
echo
echo "=== Setup Complete ==="
echo
echo "Bucket: s3://$BUCKET_NAME"
echo "Region: $AWS_REGION"
echo "CI User: $CI_USER (read-only)"
echo "Admin User: $ADMIN_USER (read/write)"
echo
echo "Next steps:"
echo "1. Upload your cache: aws s3 sync ~/.cache/huggingface s3://$BUCKET_NAME/cache/"
echo "2. Add AWS credentials to GitHub Secrets"
echo "3. Update bucket name in .github/workflows/build.yml"
echo "4. Uncomment S3 sync steps in workflow"
