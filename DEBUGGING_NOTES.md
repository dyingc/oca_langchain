# Debugging Notes - Message Validation

## Critical Bug: Anthropic Validation Modifying Valid Sequences

### Test That Fails
```python
def test_anthropic_normal_sequence():
    messages = [
        AnthropicMessage(role="user", content="Search the web"),
        AnthropicMessage(role="assistant", content=[
            AnthropicContentBlock(type="tool_use", id="toolu_123", name="search", input={"query": "test"}),
            AnthropicContentBlock(type="tool_use", id="toolu_456", name="search", input={"query": "test2"}),
        ]),
        AnthropicMessage(role="user", content=[
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_123", content=json.dumps({"result": "ok"})),
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_456", content=json.dumps({"result": "ok2"})),
        ]),
        AnthropicMessage(role="user", content="What did you find?"),
    ]
    
    result = validate_and_fix_anthropic_message_sequence(messages)
    # Should be unchanged, but Message 1 role changes from "assistant" to "user"!
```

### What to Check in `validate_and_fix_anthropic_message_sequence()`:

1. **Line-by-line comparison needed**: Add debug prints to see:
   - Which messages are being processed
   - Which conditions are being triggered
   - Why assistant message is becoming user message

2. **Check the condition at line ~260-280**:
   ```python
   # Look ahead to check if we have all required tool_results
   j = i + 1
   found_tool_results = 0
   valid_sequence = True
   
   while j < len(anthropic_messages) and found_tool_results < num_tool_uses:
       next_msg = anthropic_messages[j]
       
       # Check if next message is a user message with tool_result blocks
       if next_msg.role == "user" and isinstance(next_msg.content, list):
           tool_result_blocks = [b for b in next_msg.content if b.type == "tool_result"]
           if tool_result_blocks:
               found_tool_results += 1
               j += 1
           else:
               valid_sequence = False
               break
   ```

   **Potential Issue**: The condition might be incorrectly setting `valid_sequence = False`

3. **Check the "Invalid sequence" handling** (lines ~280-310):
   ```python
   else:
       # Invalid sequence: remove tool_use blocks from assistant message
       text_blocks = [b for b in msg.content if b.type == "text"]
       if text_blocks:
           text_content = "\n".join([b.text or "" for b in text_blocks])
           cleaned_messages.append(
               AnthropicMessage(role=msg.role, content=text_content)
           )
   ```
   
   **Potential Issue**: When `valid_sequence` is incorrectly False, this code path converts list content to string content, which might explain why the message changes.

## Medium Priority Bug: OpenAI Orphaned Tool Handling

### Test That Fails
```python
def test_openai_interrupted_by_user():
    messages = [
        ChatMessage(role="user", content="Search the web"),
        ChatMessage(role="assistant", content="", tool_calls=[call_123, call_456]),
        ChatMessage(role="user", content="Wait, stop!"),  # INTERRUPT
        ChatMessage(role="tool", tool_call_id="call_123", content="..."),
        ChatMessage(role="tool", tool_call_id="call_456", content="..."),
    ]
    
    result = validate_and_fix_message_sequence(messages)
    # Expected: 3 messages
    # Got: 5 messages (tool messages not removed!)
```

### What to Check in `validate_and_fix_message_sequence()`:

1. **Check lines 194-200** (orphaned tool skipping):
   ```python
   # Skip orphaned tool responses
   i += 1
   while i < len(messages) and messages[i].role == "tool":
       logger.warning(
           f"[MESSAGE VALIDATION] Skipping orphaned tool response at message {i}"
       )
       i += 1
   ```

2. **Add debug output** to see:
   - What is the value of `i` when entering the while loop?
   - How many tool messages are being skipped?
   - Are the tool messages actually being reached?

3. **Check the logic flow**:
   - Line 192: Assistant message (without tool_calls) is added
   - Line 195: `i += 1` moves to next message
   - Line 196: Should skip all consecutive tool messages
   - But test shows tool messages are still in result

**Hypothesis**: Maybe the while loop condition is not being met, or `i` is not being updated correctly after the loop.

## Debugging Commands

### Add Debug Prints

In `validate_and_fix_anthropic_message_sequence()`:
```python
def validate_and_fix_anthropic_message_sequence(anthropic_messages):
    cleaned_messages = []
    i = 0
    
    print(f"\n=== DEBUG: Starting validation with {len(anthropic_messages)} messages ===")
    
    while i < len(anthropic_messages):
        msg = anthropic_messages[i]
        print(f"\n[{i}] Processing: role={msg.role}, has_tool_use={isinstance(msg.content, list) and any(b.type=='tool_use' for b in msg.content if isinstance(msg.content, list))}")
        
        # ... rest of code with more prints
```

### Run Single Test
```python
# In test_message_validation.py, comment out all tests except:
if __name__ == "__main__":
    test_anthropic_normal_sequence()  # Or any failing test
```

### Check Message Objects
```python
# Add this in validation function
import inspect
print(f"Message object type: {type(msg)}")
print(f"Message attributes: {dir(msg)}")
print(f"Message role: {msg.role}")
print(f"Message content type: {type(msg.content)}")
print(f"Message content: {msg.content}")
```

## Files to Modify

1. `/Users/yingdong/VSCode/oca_langchain/converters/anthropic_request_converter.py`
   - Function: `validate_and_fix_anthropic_message_sequence()`
   - Lines: ~230-340

2. `/Users/yingdong/VSCode/oca_langchain/api.py`
   - Function: `validate_and_fix_message_sequence()`
   - Lines: ~139-206

## Quick Verification Commands

```bash
# Run single test
python -c "
import sys
sys.path.insert(0, '.')
from test_message_validation import test_anthropic_normal_sequence
test_anthropic_normal_sequence()
"

# Check if valid sequence is preserved
python -c "
import sys, json
sys.path.insert(0, '.')
from api import validate_and_fix_message_sequence, ChatMessage

messages = [
    ChatMessage(role='user', content='Hello'),
    ChatMessage(role='assistant', content='Hi'),
    ChatMessage(role='user', content='How are you?'),
]

result = validate_and_fix_message_sequence(messages)
print(f'Input: {len(messages)} messages')
print(f'Output: {len(result)} messages')
print(f'Same object: {messages == result}')
for i, (m1, m2) in enumerate(zip(messages, result)):
    print(f'  [{i}] Same: {m1.role == m2.role and m1.content == m2.content}')
"
```

