# LangGraph Agent Result Format

## Cấu trúc `result`

`app.invoke()` trả về một `dict` với key `messages` chứa danh sách message theo thứ tự hội thoại:

```python
{
    "messages": [
        HumanMessage(...),
        AIMessage(...),      # có thể có tool_calls
        ToolMessage(...),    # nếu tool được gọi
        AIMessage(...)       # câu trả lời cuối
    ]
}
```

## Các loại Message

### HumanMessage
Câu hỏi/input từ user.
```python
HumanMessage(
    content="What's the weather in San Francisco?"
)
```

### AIMessage (gọi tool)
LLM quyết định cần gọi tool, `content` thường rỗng.
```python
AIMessage(
    content="",
    tool_calls=[{
        "id": "call_xxx",
        "name": "get_weather",
        "args": {"city": "San Francisco"}
    }]
)
```

### ToolMessage
Kết quả trả về từ tool sau khi thực thi.
```python
ToolMessage(
    content="It's always sunny in San Francisco!",
    tool_call_id="call_xxx",
    name="get_weather"
)
```

### AIMessage (final)
LLM tổng hợp và trả lời cuối cùng dựa trên kết quả tool.
```python
AIMessage(
    content="The weather in San Francisco is sunny!"
)
```

## Luồng hoạt động

| Bước | Message type | Ý nghĩa |
|------|-------------|---------|
| 1 | `HumanMessage` | Câu hỏi của user |
| 2 | `AIMessage` (tool_calls) | LLM quyết định gọi tool nào, với args gì |
| 3 | `ToolMessage` | Kết quả trả về từ tool |
| 4 | `AIMessage` (final) | LLM tổng hợp và trả lời cuối cùng |

> Nếu câu hỏi không cần tool: chỉ có bước 1 và 4.
> Nếu cần nhiều tool: bước 2–3 lặp lại trước khi ra bước 4.

---

# LangGraph Graph Configuration

## StateGraph

```python
graph = StateGraph(MessagesState)
```

Tạo một graph với **state** là `MessagesState` — built-in state chứa danh sách messages. Mỗi node nhận state hiện tại, xử lý, và trả về phần state muốn cập nhật (append vào `messages`).

Có thể tự định nghĩa state riêng thay vì dùng `MessagesState`:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str  # thêm field tuỳ ý

graph = StateGraph(MyState)
```

---

## Nodes

Node là một **bước xử lý** trong graph, nhận state và trả về state mới.

```python
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
```

| Node | Hàm xử lý | Vai trò |
|------|-----------|---------|
| `"agent"` | `call_model` | Gọi LLM, sinh ra response hoặc tool_calls |
| `"tools"` | `ToolNode(tools)` | Thực thi tool mà LLM yêu cầu |

`ToolNode` là built-in, tự động đọc `tool_calls` từ message cuối và thực thi tool tương ứng.

---

## Edges (cạnh cố định)

Luôn đi từ node này sang node kia, không điều kiện.

```python
graph.add_edge(START, "agent")   # bắt đầu → agent
graph.add_edge("tools", "agent") # sau khi tools xong → quay lại agent
```

- `START` — điểm vào của graph (built-in)
- `END` — điểm kết thúc của graph (built-in)

---

## Conditional Edges (cạnh có điều kiện)

Sau một node, gọi hàm router để quyết định đi tiếp đâu.

```python
graph.add_conditional_edges("agent", should_continue, ["tools", END])
```

| Tham số | Ý nghĩa |
|---------|---------|
| `"agent"` | Node nguồn |
| `should_continue` | Hàm router, nhận state và trả về tên node tiếp theo |
| `["tools", END]` | Danh sách các đích hợp lệ (tuỳ chọn, để validate) |

Ví dụ hàm router:

```python
def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

---

## Sơ đồ luồng

```
START
  │
  ▼
[agent] ──(có tool_calls)──▶ [tools]
  │                              │
  │ (không có tool_calls)        │
  ▼                              │
 END        ◀────────────────────┘
```

---

## Compile và chạy

```python
app = graph.compile()

# Chạy đồng bộ
result = app.invoke({"messages": [("user", "câu hỏi")]})

# Chạy streaming (từng bước)
for step in app.stream({"messages": [("user", "câu hỏi")]}):
    print(step)
```

`compile()` có thể nhận thêm tùy chọn:

```python
app = graph.compile(
    checkpointer=...,        # lưu state để resume sau
    interrupt_before=["tools"],  # dừng trước khi vào node "tools"
)
```

---

## Truy cập dữ liệu

```python
# Câu trả lời cuối
result["messages"][-1].content

# Danh sách tất cả messages
result["messages"]

# Tool call đầu tiên (tên tool + args)
result["messages"][1].tool_calls[0]["name"]
result["messages"][1].tool_calls[0]["args"]

# Kết quả từ tool
result["messages"][2].content
```

---

## Tài liệu tham khảo

### LangGraph
| Chủ đề | Link |
|--------|------|
| Quickstart | https://langchain-ai.github.io/langgraph/tutorials/introduction/ |
| StateGraph API | https://langchain-ai.github.io/langgraph/reference/graphs/ |
| MessagesState | https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.MessagesState |
| ToolNode | https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.ToolNode |
| How-to: Tool calling | https://langchain-ai.github.io/langgraph/how-tos/tool-calling/ |
| How-to: Streaming | https://langchain-ai.github.io/langgraph/how-tos/streaming/ |
| How-to: Persistence (checkpointer) | https://langchain-ai.github.io/langgraph/how-tos/persistence/ |
| Conceptual Guide | https://langchain-ai.github.io/langgraph/concepts/ |

### LangChain
| Chủ đề | Link |
|--------|------|
| Tool calling | https://python.langchain.com/docs/concepts/tool_calling/ |
| `@tool` decorator | https://python.langchain.com/docs/concepts/tools/ |
| ChatOpenAI (OpenAI-compatible) | https://python.langchain.com/docs/integrations/chat/openai/ |
| Message types | https://python.langchain.com/docs/concepts/messages/ |

### Qwen API
| Chủ đề | Link |
|--------|------|
| Qwen Cloud (international) | https://home.qwencloud.com |
| Model list | https://home.qwencloud.com/models |
| OpenAI-compatible API docs | https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api |
