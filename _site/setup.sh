echo "USING RUBY VERSION: "$(ruby --version)
sudo gem install bundler
bundle install
bundle exec jekyll serve