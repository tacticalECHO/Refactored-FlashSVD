# Contributing to FlashSVD

Thank you for your interest in contributing to FlashSVD! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Issues

- **Bug reports**: Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Feature requests**: Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- **Questions**: Use [GitHub Discussions](https://github.com/Zishan-Shao/FlashSVD/discussions)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test your changes**: Run the test suite
5. **Commit with clear messages**: Follow conventional commits format
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**

## 🧪 Testing

Before submitting a PR, please run:

```bash
# Run test suite
cd test/scripts/
./test_all_methods.sh

# Test your specific changes
flashsvd compress --model bert-base-uncased --task sst2 --method <your_method>
flashsvd eval --checkpoint <checkpoint_path> --task sst2
```

## 📝 Code Style

- **Python**: Follow PEP 8 guidelines
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type hints to function signatures
- **Comments**: Write clear, concise comments

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

1. **New Compression Methods**
   - SVD-LLM, Dobi-SVD, etc.
   - Implementation in `src/flashsvd/compression/`

2. **Model Architectures**
   - Qwen, Mistral, Falcon, etc.
   - Add architecture detection in `src/flashsvd/compress.py`

3. **Custom Dataset Support**
   - CSV/JSON dataset loading
   - Implementation in `src/flashsvd/finetune/`

4. **Performance Optimizations**
   - Kernel improvements
   - Memory optimizations

5. **Documentation**
   - Tutorials and examples
   - API documentation

## 📧 Contact

For questions about contributing, contact:
- Email: zs89@duke.edu
- GitHub Discussions: [FlashSVD Discussions](https://github.com/Zishan-Shao/FlashSVD/discussions)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FlashSVD! 🎉
