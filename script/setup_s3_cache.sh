#!/bin/bash
# Script to set up S3 bucket for HuggingFace model cache

set -e

BUCKET_NAME="red-candle-hf-cache"
CACHE_DIR="$HOME/.cache/huggingface"
BACKUP_DIR="$HOME/.cache/huggingface.backup"
MINIMAL_CACHE="/tmp/minimal_hf_cache"

echo "=== S3 Cache Setup for Red Candle ==="
echo

# Step 1: Backup existing cache
if [ -d "$CACHE_DIR" ]; then
    echo "1. Backing up existing cache..."
    if [ -d "$BACKUP_DIR" ]; then
        echo "   Backup already exists at $BACKUP_DIR"
    else
        cp -r "$CACHE_DIR" "$BACKUP_DIR"
        echo "   Backed up to $BACKUP_DIR"
    fi
else
    echo "1. No existing cache found"
fi

# Step 2: Identify models in cache
echo
echo "2. Cache preparation..."
echo "   Option A: Clear cache and run tests (downloads models)"
echo "   Option B: Use existing cache as-is (no downloads)"
echo
read -p "Choose option [A/B]: " option

if [ "$option" = "A" ] || [ "$option" = "a" ]; then
    # Clear and rebuild
    echo "   Clearing cache..."
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"

    echo "   Running specs to populate cache..."
    bundle exec rspec || true  # Don't fail if some specs fail

elif [ "$option" = "B" ] || [ "$option" = "b" ]; then
    # Use existing cache
    echo "   Using existing cache as-is"
    echo "   Current cache statistics:"
    du -sh "$CACHE_DIR" 2>/dev/null || echo "No cache found"
fi

# Step 3: Get cache size
echo
echo "3. Cache statistics:"
if [ -d "$CACHE_DIR/hub" ]; then
    echo "   Total size: $(du -sh $CACHE_DIR | cut -f1)"
    echo "   Model count: $(ls -d $CACHE_DIR/hub/models--* 2>/dev/null | wc -l)"
    echo "   Models:"
    for model_dir in $CACHE_DIR/hub/models--*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g')
            size=$(du -sh "$model_dir" | cut -f1)
            echo "     - $model_name ($size)"
        fi
    done
fi

# Step 4: Upload to S3
echo
echo "4. Upload to S3"
read -p "Upload cache to s3://$BUCKET_NAME/? [y/N]: " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "   Creating bucket if needed..."
    aws s3 mb "s3://$BUCKET_NAME" --region us-east-1 2>/dev/null || true

    echo "   Uploading cache..."
    aws s3 sync "$CACHE_DIR" "s3://$BUCKET_NAME/cache/" \
        --delete \
        --exclude ".DS_Store" \
        --exclude "*.pyc" \
        --exclude "__pycache__/*"

    echo "   Upload complete!"
    echo
    echo "   S3 URL: s3://$BUCKET_NAME/cache/"
    echo "   Size: $(aws s3 ls --summarize --recursive s3://$BUCKET_NAME/cache/ | grep "Total Size" | cut -d: -f2)"
fi

# Step 5: Generate GitHub Actions snippet
echo
echo "5. GitHub Actions Configuration"
echo "   Add this to your workflow:"
echo
cat << 'EOF'
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Sync HuggingFace cache from S3
        run: |
          echo "Downloading HuggingFace cache from S3..."
          aws s3 sync s3://BUCKET_NAME/cache/ ~/.cache/huggingface/ --quiet
          echo "Cache size: $(du -sh ~/.cache/huggingface | cut -f1)"
          echo "Models cached: $(ls -d ~/.cache/huggingface/hub/models--* 2>/dev/null | wc -l)"

          # Set offline mode to prevent any API calls
          export HF_HUB_OFFLINE=1
EOF

echo
echo "   Don't forget to:"
echo "   - Replace BUCKET_NAME with: $BUCKET_NAME"
echo "   - Add AWS_ACCESS_KEY_ID to GitHub secrets"
echo "   - Add AWS_SECRET_ACCESS_KEY to GitHub secrets"
echo "   - Consider using IAM role instead of keys for better security"

# Step 6: Restore option
echo
echo "6. To restore your original cache:"
echo "   mv $BACKUP_DIR $CACHE_DIR"