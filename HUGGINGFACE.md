# Huggingface Authentication

Many models from [🤗 Huggingface](https://huggingface.co) are freely availble. For instance our default reranker model (`cross-encoder/ms-marco-MiniLM-L-12-v2`) and our default embedding models (`jinaai/jina-embeddings-v2-base-en` and `sentence-transformers/all-MiniLM-L6-v2`) are freely available without authentication. However, certain models, transformers from [MistralAI](https://huggingface.co/mistralai) such as `mistralai/Mistral-7B-Instruct-v0.1`, require you consent to the author's terms of service, and that requires authentication.

## If you are unauthenticated

If you are unauthenticated and try to access a model that requires authentication, you will see an error like this:
```
red-candle/lib/candle/llm.rb:32:in `_from_pretrained': Failed to load model: Failed to download config: request error: HTTP status client error (401 Unauthorized) for url (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json) (RuntimeError)
	from /Users/robert/Documents/red-candle/lib/candle/llm.rb:32:in `from_pretrained'
	from (irb):18:in `<main>'
	from <internal:kernel>:187:in `loop'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/gems/3.3.0/gems/irb-1.13.1/exe/irb:9:in `<top (required)>'
	from /Users/robert/.rbenv/versions/3.3.8/bin/irb:25:in `load'
	from /Users/robert/.rbenv/versions/3.3.8/bin/irb:25:in `<top (required)>'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli/exec.rb:59:in `load'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli/exec.rb:59:in `kernel_load'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli/exec.rb:23:in `run'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli.rb:452:in `exec'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/vendor/thor/lib/thor/command.rb:28:in `run'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/vendor/thor/lib/thor/invocation.rb:127:in `invoke_command'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/vendor/thor/lib/thor.rb:538:in `dispatch'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli.rb:35:in `dispatch'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/vendor/thor/lib/thor/base.rb:584:in `start'
	from /Users/robert/.rbenv/versions/3.3.8/lib/ruby/site_ruby/3.3.0/bundler/cli.rb:29:in `start'
	... 5 levels...
irb(main):020> 
```

## To authenticate

You have two options:

### Option 1: Huggingface CLI

You can authenticate by running the following command:
```sh
huggingface login
```

### Option 2:Create a token file

You can skip the Huggingface CLI and create a token file manually. Create a file at `$HOME/.cache/huggingface/token` and populate it with your token.

```sh
echo 'hf_your_token_here' > $HOME/.cache/huggingface/token
```

Once you've authenticated and consented to the terms of the model, you should be able to use it.