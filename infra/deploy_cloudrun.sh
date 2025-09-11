#!/usr/bin/env bash
set -euo pipefail
IMG=gcr.io/$GOOGLE_CLOUD_PROJECT/trialwhisperer
REGION=${REGION:-europe-west1}

gcloud builds submit --tag $IMG
gcloud run deploy trialwhisperer \
  --image $IMG \
  --region $REGION \
  --allow-unauthenticated \
  --platform managed \
  --max-instances 1