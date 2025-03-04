#!/usr/bin/env sh
curl -s -X POST https://api.flower.blue/v1/organization/projects/1/api_keys \
-H "Authorization: Bearer $FI_MGMT_API_KEY" \
-H "Content-Type: application/json" \
-d '{
  "billing_id": "test_billing_id_123",
  "name": "My Test API Key",
  "total_tokens_limit": 1000000
}' | jq -r '.api_key'
