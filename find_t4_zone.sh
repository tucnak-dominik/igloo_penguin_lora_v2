#!/bin/bash

PROJECT_ID=$(gcloud config get-value project)
ZONES=("us-central1-a" "us-central1-b" "us-west1-b" "us-east1-d" "europe-west1-b" "europe-west4-a")

for ZONE in "${ZONES[@]}"; do
  echo "🌍 Zkouším zónu: $ZONE"
  gcloud compute instances create test-gpu-check \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=10GB \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform \
    --no-address \
    --quiet &> /dev/null

  if [ $? -eq 0 ]; then
    echo "✅ VOLNÁ zóna: $ZONE"
    gcloud compute instances delete test-gpu-check --zone=$ZONE --quiet
    break
  else
    echo "❌ Zóna $ZONE nedostupná"
  fi
done
