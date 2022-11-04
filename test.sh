curl https://api.openai.com/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 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" \
-d '{"model": "text-davinci-002", "prompt": "Say this is a test", "temperature": 0, "max_tokens": 6}'