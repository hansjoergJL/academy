# Academy Agents - Development Guidelines

## Coding Quality Requirements

### **Phase Development Protocol**
Before testing any trainer or code change, follow this sequence:

1. **Code Implementation**
   - Write/update the code
   - Run linter and type checks
   - Fix syntax errors immediately

2. **Pre-Test Validation**
   - `python -m py_flakes <module.py>`
   - `python -m mypy <module.py>`
   - Check imports and basic syntax

3. **Dependency Check**
   - Verify all required packages are installed
   - Test imports in isolation
   - Handle version compatibility

4. **Test Execution**
   - Run with debug flags first
   - Start with small datasets
   - Monitor for warnings/errors

5. **Code Review**
   - Check for deprecated parameters
   - Validate function signatures
   - Review error handling

## **Current Known Issues**

### **ModelTrainer Status**
- **Working**: Basic LoRA setup, model loading, dataset preparation
- **Issues**: 
  - `evaluation_strategy` parameter deprecated
  - Dataset hashing warnings
  - Device multi-placement errors
  - `dtype` parameter deprecation warnings

### **Immediate Action Items**
1. Fix TrainingArguments parameters
2. Resolve dataset caching warnings
3. Implement proper error handling
4. Add comprehensive testing

### **Phase Development Protocol**
Before testing any trainer or code change, follow this sequence:

1. **Code Implementation**
   - Write/update the code
   - Run linter and type checks: `python -m pyflakes <module.py>` and `python -m mypy <module.py>`
   - Fix syntax errors immediately
   - Fix import path issues
   - Ensure correct indentation
   - Verify function signatures

2. **Pre-Test Validation**
   - Test imports in isolation
   - Check dependencies compatibility
   - Test with PYTHONPATH setup

3. **Dependency Check**
   - Verify all required packages are installed
   - Handle version compatibility

4. **Test Execution**
   - Create simple test cases for each module
   - Run with debug flags first
   - Start with small datasets
   - Monitor for warnings/errors
   - Use `python academy/tests/test_modules.py --test all`

5. **Code Review**
   - Check for deprecated parameters
   - Validate function signatures
   - Review error handling

### **Testing Requirements**
- Always test with `--train cl` option first
- Use small datasets for initial testing
- Monitor memory usage on CPU
- Validate model saving/loading
- Create comprehensive test cases before training

### **Quality Gates**
No code testing unless:
- [ ] Syntax check passes: `python -m pyflakes academy/<module>.py`
- [ ] Import validation successful: `python -c "import academy.<module>"`
- [ ] Dependencies verified: All packages installable
- [ ] Basic functionality tested: `python academy/tests/test_modules.py --test all`
- [ ] Training pipeline functional: End-to-end test successful

### **Version Control Workflow**
1. Implement changes
2. Run quality checks: `python -m pyflakes academy/*.py` and `python -m mypy academy/*.py --ignore-missing-imports`
3. Test locally: `python academy/tests/test_modules.py --test all`
4. Commit with detailed message
5. Push only after validation

---

*Last Updated: Comprehensive Testing Protocol Added*