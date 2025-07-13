unit LibChatLLM;

{$IFDEF FPC}{$MODE DELPHI}{$ENDIF}

interface

uses
  SysUtils, Classes
  {$ifdef dcc}
  , System.Threading
  {$endif}
  ;

const
  CHATLLMLIB = 'libchatllm';

type

  TPrintType = (
      PRINT_CHAT_CHUNK        = 0,
      // below items share the same value with BaseStreamer::TextType
      PRINTLN_META            = 1,    // print a whole line: general information
      PRINTLN_ERROR           = 2,    // print a whole line: error message
      PRINTLN_REF             = 3,    // print a whole line: reference
      PRINTLN_REWRITTEN_QUERY = 4,    // print a whole line: rewritten query
      PRINTLN_HISTORY_USER    = 5,    // print a whole line: user input history
      PRINTLN_HISTORY_AI      = 6,    // print a whole line: AI output history
      PRINTLN_TOOL_CALLING    = 7,    // print a whole line: tool calling (supported by only a few models)
      PRINTLN_EMBEDDING       = 8,    // print a whole line: embedding (example: "0.1,0.3,...")
      PRINTLN_RANKING         = 9,    // print a whole line: ranking (example: "0.8")
      PRINTLN_TOKEN_IDS       =10,    // print a whole line: token ids (example: "1,3,5,8,...")
      PRINTLN_LOGGING         =11,    // print a whole line: internal logging with the first char indicating level
                                      // (space): None; D: Debug; I: Info; W: Warn; E: Error; .: continue
      PRINTLN_BEAM_SEARCH     =12,    // print a whole line: a result of beam search with a prefix of probability
                                      // (example: "0.8,....")
      PRINTLN_MODEL_INFO      =13,    // when a model is started, print a whole line of basic model information (json format)
                                      // (example: {"name": "llama", "context_length": 100, "capabilities": [text, ...], ...})
      PRINT_THOUGHT_CHUNK     =14,    // same as PRINT_CHAT_CHUNK, but this from "thoughts".
                                      // possible leading or trailing tags (such as <think>, </think>) are removed.
                                      // use `+detect_thoughts` to enable this.

      PRINT_EVT_ASYNC_COMPLETED       = 100,   // last async operation completed (utf8_str is null)
      PRINT_EVT_THOUGHT_COMPLETED     = 101    // thought completed
  );

  TEmbeddingPurpose = (
    epForDoc   = 0,    // for document
    epForQuery = 1     // for query
  );

  TChatLLMPrint = procedure(UserData: Pointer; APrintType: Integer; AUTF8Str: PAnsiChar); cdecl;
  TChatLLMEnd = procedure(UserData: Pointer); cdecl;

  PChatLLMObj = Pointer;

  {
    @brief create ChatLLM object

    @return                  the object
  }
  function ChatLLMCreate: PChatLLMObj; stdcall; external CHATLLMLIB name 'chatllm_create';

  {
    @brief append a command line option

    @param Obj              Model object
    @param AUTF8Str         A command line option
  }
  procedure ChatLLMAppendParam(Obj: PChatLLMObj; AUTF8Str: PAnsiChar); stdcall; external CHATLLMLIB name 'chatllm_append_param';

  {
    @brief start the model

    @param Obj              Model object
    @param FPrint           Callback function for printing
    @param FEnd             Callback function when model generation ends
    @param UserData         User data provided to callback functions
    @return                 0 if succeeded
  }
  function ChatLLMStart(Obj: PChatLLMObj; FPrint: TChatLLMPrint; FEnd: TChatLLMEnd; UserData: Pointer): Integer; stdcall; external CHATLLMLIB name 'chatllm_start';

  {
    @brief set max number of generated tokens in a new round of conversation

    @param[in] obj               model object
    @param[in] gen_max_tokens    -1 for as many as possible
  }
  procedure ChatLLMSetGenMaxTokens(Obj: PChatLLMObj; GenMaxTokens: Integer); stdcall; external CHATLLMLIB name 'chatllm_set_gen_max_tokens';

  {
    @brief restart (i.e. discard history)

    * When a session has been loaded, the model is restarted to the point that the session is loaded;

        Note: this would not work if `--extending` is not `none` or the model uses SWA.

    * Otherwise, it is restarted from the very beginning.

    @param[in] obj               model object
    @param[in] AUTF8Str          update to a new system prompt
 *                               if nil, then system prompt is kept unchanged.
  }
  procedure ChatLLMRestart(Obj: PChatLLMObj; AUTF8Str: PAnsiChar); stdcall; external CHATLLMLIB name 'chatllm_restart';

  {
    @brief prepare to generate a multimedia input, i.e. clear previously added pieces.

    Each `chatllm_obj` has a global multimedia message object, which can be used as user input,
    or chat history, etc.

    @param[in] obj               model object
    @return                      0 if succeeded
  }
  procedure ChatLLMMultimediaMsgPrepare(Obj: PChatLLMObj); stdcall; external CHATLLMLIB name 'chatllm_multimedia_msg_prepare';

  {
    @brief add a piece to a multimedia message

    Remember to clear the message by `chatllm_multimedia_msg_prepare` when starting a new message.

    @param[in] obj               model object
    @param[in] type              type ::= "text" | "image" | "video" | "audio" | ...
    @param[in] utf8_str          content, i.e. utf8 text content, or base64 encoded data of multimedia data.
    @return                      0 if succeeded
  }
  function ChatLLMMultimediaMsgAppend(Obj: PChatLLMObj; _Type: PAnsiChar; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_multimedia_msg_append';

  {
    @brief user input

    This function is synchronized, i.e. it returns after model generation ends and `f_end` is called.

    @param[in] obj               model object
    @param[in] utf8_str          user input
    @return                      0 if succeeded
  }
  function ChatLLMUserInput(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_user_input';

  {
    @brief take current multimedia message as user input and run

    This function is synchronized, i.e. it returns after model generation ends and `f_end` is called.

    @param[in] obj               model object
    @return                      0 if succeeded
  }
  function ChatLLMUserInputMultimediaMsg(Obj: PChatLLMObj): Integer; stdcall; external CHATLLMLIB name 'chatllm_user_input_multimedia_msg';

  {
    @brief set prefix for AI generation

    This prefix is used in all following rounds..

    @param[in] obj               model object
    @param[in] utf8_str          prefix
    @return                      0 if succeeded
  }
  function ChatLLMSetAIPrefix(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_set_ai_prefix';

  {
    @brief tool input

    - If this function is called before `chatllm_user_input` returns, this is asynchronized,
    - If this function is called after `chatllm_user_input` returns, this is equivalent to
      `chatllm_user_input`.

    @param[in] obj               model object
    @param[in] utf8_str          user input
    @return                      0 if succeeded
  }
  function ChatLLMToolInput(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_tool_input';

  {
    @brief feed in text generated by external tools

    This text is treated as part of AI's generation. After this is called, LLM generation
    is continued.

    Example:

    ```c
    // in `f_print` callback:
    chatllm_abort_generation();
    chatllm_tool_completion(...);
    ```

    @param[in] obj               model object
    @param[in] utf8_str          text
    @return                      0 if succeeded
   }
  function ChatLLMToolCompletion(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_tool_completion';

  {
    @brief tokenize

    token ids are emitted through `PRINTLN_TOKEN_IDS`.

    @param[in] obj               model object
    @param[in] utf8_str          text
    @return                      number of ids if succeeded. otherwise -1.
  }
  function ChatLLMTextTokenize(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_text_tokenize';

  {
    @brief text embedding

    embedding is emitted through `PRINTLN_EMBEDDING`.

    @param[in] obj               model object
    @param[in] utf8_str          text
    @param[in] purpose           purpose, see `EmbeddingPurpose`
    @return                      0 if succeeded
  }
  function ChatLLMTextEmbedding(Obj: PChatLLMObj; AUTF8Str: PAnsiChar; Purpose: Integer): Integer; stdcall; external CHATLLMLIB name 'chatllm_text_embedding';

  {
    @brief question & answer ranking

    embedding is emit through `PRINTLN_RANKING`.

    @param[in] obj               model object
    @param[in] utf8_str_q        question
    @param[in] utf8_str_a        answer
    @return                      0 if succeeded
  }
  function ChatLLMQARank(Obj: PChatLLMObj; AUTF8StrQ, AUTF8StrA: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_qa_rank';

  {
    @brief switching RAG vector store

    @param[in] obj               model object
    @param[in] name              vector store name
    @return                      0 if succeeded
  }
  function ChatLLMRAGSelectStore(Obj: PChatLLMObj; AName: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_rag_select_store';

  {
    @brief abort generation

    This function is asynchronized, i.e. it returns immediately.

    @param[in] obj               model object
  }
  procedure ChatllmAbortGeneration(Obj: PChatLLMObj); stdcall; external CHATLLMLIB name 'chatllm_abort_generation';

  {
    @brief show timing statistics

    Result is sent to `f_print`.

    @param[in] obj               model object
  }
  procedure ChatLLMShowStatistics(Obj: PChatLLMObj); stdcall; external CHATLLMLIB name 'chatllm_show_statistics';

  {
    @brief save current session on demand

    Note: Call this from the same thread of `chatllm_user_input()`.

    If chat history is empty, then system prompt is evaluated and saved.

    @param[in] obj               model object
    @param[in] utf8_str          file full name
    @return                      0 if succeeded
  }
  function ChatLLMSaveSession(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_save_session';

  {
    @brief load a session on demand

    Note: Call this from the same thread of `chatllm_user_input()`.

    @param[in] obj               model object
    @param[in] utf8_str          file full name
    @return                      0 if succeeded
  }
  function ChatLLMLoadSession(Obj: PChatLLMObj; AUTF8Str: PAnsiChar): Integer; stdcall; external CHATLLMLIB name 'chatllm_load_session';

type
  TLLMPrintEvent = procedure (Sender: TObject; S: string) of object;
  TLLMStateChangedEvent = procedure (Sender: TObject; ABusy: Boolean) of object;

  TLLMTextEmbeddingResult = procedure (Sender: TObject; AState: Integer; AEmbedding: array of Single) of object;
  TLLMQARankingResult = procedure (Sender: TObject; AState: Integer; ARanking: Single) of object;

  { TChatLLM }

  TChatLLM = class
  private
    FObj: PChatLLMObj;
    FOnThoughtChunk: TLLMPrintEvent;
    FOnThoughtEnded: TNotifyEvent;
    FOutputAcc: string;
    FThoughtAcc: string;
    FReferences: TStringList;
    FMetaAcc: string;
    FMiscResult: string;
    FOnPrintHistoryAI: TLLMPrintEvent;
    FOnPrintRewrittenQuery: TLLMPrintEvent;
    FOnPrintToolCalling: TLLMPrintEvent;
    FOnPrintError: TLLMPrintEvent;
    FOnChunk: TLLMPrintEvent;
    FOnPrintMeta: TLLMPrintEvent;
    FOnPrintReference: TLLMPrintEvent;
    FOnPrintHistoryUser: TLLMPrintEvent;
    FOnGenerationEnded: TNotifyEvent;
    FBusy: Integer;
    FOnStateChanged: TLLMStateChangedEvent;
    FOnTextEmbeddingResult: TLLMTextEmbeddingResult;
    FOnQARankingResult: TLLMQARankingResult;
    FAutoAbortSufffix: string;
    FCallingMode: Boolean;
    FCallResult: TLLMPrintEvent;
    FModelInfo: string;
    function GetBusy: Boolean;
    procedure SetOnChunk(const Value: TLLMPrintEvent);
    procedure SetOnPrintError(const Value: TLLMPrintEvent);
    procedure SetOnPrintHistoryAI(const Value: TLLMPrintEvent);
    procedure SetOnPrintHistoryUser(const Value: TLLMPrintEvent);
    procedure SetOnPrintMeta(const Value: TLLMPrintEvent);
    procedure SetOnPrintReference(const Value: TLLMPrintEvent);
    procedure SetOnPrintRewrittenQuery(const Value: TLLMPrintEvent);
    procedure SetOnPrintToolCalling(const Value: TLLMPrintEvent);
    procedure SetGenMaxTokens(const Value: Integer);
    procedure SetOnGenerationEnded(const Value: TNotifyEvent);
    procedure SetOnStateChanged(const Value: TLLMStateChangedEvent);
    procedure SetBusy(AValue: Boolean);
    procedure SetOnQARankingResult(const Value: TLLMQARankingResult);
    procedure SetOnTextEmbeddingResult(const Value: TLLMTextEmbeddingResult);

    procedure ChatEnd(AState: Integer);
    procedure CallChatEnd(AState: Integer);
    procedure TextEmbeddingEnd(AState: Integer);
    procedure QARankingEnd(AState: Integer);
    procedure SetAIPrefix(const Value: string);

  protected
    FThinking: Boolean;
    procedure InternalChunk(S: string);
  protected
    procedure DoBeforeChat; virtual;
    procedure HandlePrint(APrintType: Integer; S: string); virtual;
    procedure HandleEnd; virtual;
  public
    constructor Create;
    destructor Destroy; override;

    function Start(): Integer;
    procedure Restart; overload;
    procedure Restart(ASysPrompt: string); overload;
    procedure AddParam(AParams: array of string); overload;
    procedure AddParam(AParams: TStrings); overload;
    procedure AddParam(AParam: string); overload;

    function Chat(const AInput: string): Integer; overload;
    function ToolInput(const AInput: string): Integer;
    function ToolCompletion(const AInput: string): Integer;
    procedure AbortGeneration;

    function CallChat(const AInput: string; OnResult: TLLMPrintEvent): Integer;

    function TextEmbedding(const AText: string; const APurpose: TEmbeddingPurpose = epForDoc): Integer;
    function QARanking(const AQustion, AAnswer: string): Integer;
    function RAGSelectStore(const AName: string): Integer;

    property GenMaxTokens: Integer write SetGenMaxTokens;
    property Busy: Boolean read GetBusy;
  public
    property OnChunk: TLLMPrintEvent read FOnChunk write SetOnChunk;
    property OnThoughtChunk: TLLMPrintEvent read FOnThoughtChunk write FOnThoughtChunk;
    property OnThoughtEnded: TNotifyEvent read FOnThoughtEnded write FOnThoughtEnded;
    property OnPrintMeta: TLLMPrintEvent read FOnPrintMeta write SetOnPrintMeta;
    property OnPrintError: TLLMPrintEvent read FOnPrintError write SetOnPrintError;
    property OnPrintReference: TLLMPrintEvent read FOnPrintReference write SetOnPrintReference;
    property OnPrintRewrittenQuery: TLLMPrintEvent read FOnPrintRewrittenQuery write SetOnPrintRewrittenQuery;
    property OnPrintHistoryUser: TLLMPrintEvent read FOnPrintHistoryUser write SetOnPrintHistoryUser;
    property OnPrintHistoryAI: TLLMPrintEvent read FOnPrintHistoryAI write SetOnPrintHistoryAI;
    property OnPrintToolCalling: TLLMPrintEvent read FOnPrintToolCalling write SetOnPrintToolCalling;
    property OnGenerationEnded: TNotifyEvent read FOnGenerationEnded write SetOnGenerationEnded;

    property OnTextEmbeddingResult: TLLMTextEmbeddingResult read FOnTextEmbeddingResult write SetOnTextEmbeddingResult;
    property OnQARankingResult: TLLMQARankingResult read FOnQARankingResult write SetOnQARankingResult;

    property OnStateChanged: TLLMStateChangedEvent read FOnStateChanged write SetOnStateChanged;

    property OutputAcc: string read FOutputAcc;
    property ThoughtAcc: string read FThoughtAcc;
    property ModelInfo: string read FModelInfo;

    property AIPrefix: string write SetAIPrefix;
    property AutoAbortSufffix: string read FAutoAbortSufffix write FAutoAbortSufffix;

    property References: TStringList read FReferences;
  end;

implementation

type

  TLLMAPIEnded = procedure(AState: Integer) of object;

  { TThreadedTask }

  TThreadedTask = class
  private
    FLLM: TChatLLM;
    FNext: TLLMAPIEnded;
    FState: Integer;
  public
    constructor Create(ALLM: TChatLLM);
    procedure Start(ANext: TLLMAPIEnded);
  protected
    procedure SyncedProc;
    procedure Exec; virtual; abstract;
  end;

  { TThreadedChatTask }

  TThreadedChatTask = class(TThreadedTask)
  private
    FInput: string;
  public
    constructor Create(ALLM: TChatLLM; AInput: string);
  protected
    procedure Exec; override;
  end;

  { TThreadedEmbeddingTask }

  TThreadedEmbeddingTask = class(TThreadedTask)
  private
    FInput: string;
    FPurpose: TEmbeddingPurpose;
  public
    constructor Create(ALLM: TChatLLM; AInput: string; APurpose: TEmbeddingPurpose);
  protected
    procedure Exec; override;
  end;

  { TThreadedQATask }

  TThreadedQATask = class(TThreadedTask)
  private
    FQuestion: string;
    FAnswer: string;
  public
    constructor Create(ALLM: TChatLLM; AQuestion, AAnswer: string);
  protected
    procedure Exec; override;
  end;

  { TThreadedToolInputTask }

  TThreadedToolInputTask = class(TThreadedTask)
  private
    FInput: string;
  public
    constructor Create(ALLM: TChatLLM; AInput: string);
  protected
    procedure Exec; override;
  end;

  { TThreadedToolCompletionTask }

  TThreadedToolCompletionTask = class(TThreadedTask)
  private
    FInput: string;
  public
    constructor Create(ALLM: TChatLLM; AInput: string);
  protected
    procedure Exec; override;
  end;

  { TLLMPrintItem }

  TLLMPrintItem = class
  private
    FLLM: TChatLLM;
    FPrintType: Integer;
    FStr: string;
  public
    constructor Create(ALLM: TChatLLM; APrintType: Integer; AUTF8Str: PAnsiChar);
    procedure SyncedRun;
  end;

  { TLLMEnd }

  TLLMEnd = class
  private
    FLLM: TChatLLM;
  public
    constructor Create(ALLM: TChatLLM);
    procedure SyncedRun;
  end;

procedure _LLMPrint(UserData: Pointer; APrintType: Integer; AUTF8Str: PAnsiChar); cdecl;
var
  O: TLLMPrintItem;
begin
  O := TLLMPrintItem.Create(TChatLLM(UserData), APrintType, AUTF8Str);
  TThread.Synchronize(nil, O.SyncedRun);
end;

procedure _LLMEnd(UserData: Pointer); cdecl;
var
  O: TLLMEnd;
begin
  O := TLLMEnd.Create(TChatLLM(UserData));
  TThread.Synchronize(nil, O.SyncedRun);
end;

{ TThreadedToolCompletionTask }

constructor TThreadedToolCompletionTask.Create(ALLM: TChatLLM; AInput: string);
begin
  inherited Create(ALLM);
  FInput := AInput;
end;

procedure TThreadedToolCompletionTask.Exec;
begin
  FState := ChatLLMToolCompletion(FLLM.FObj, PUTF8Char(UTF8Encode(FInput)));
end;

{ TThreadedEmbeddingTask }

constructor TThreadedEmbeddingTask.Create(ALLM: TChatLLM; AInput: string; APurpose: TEmbeddingPurpose);
begin
  inherited Create(ALLM);
  FInput := AInput;
  FPurpose := APurpose;
end;

procedure TThreadedEmbeddingTask.Exec;
begin
  FState := ChatLLMTextEmbedding(FLLM.FObj, PUTF8Char(UTF8Encode(FInput)), Integer(FPurpose));
end;

{ TThreadedQATask }

constructor TThreadedQATask.Create(ALLM: TChatLLM; AQuestion, AAnswer: string);
begin
  inherited Create(ALLM);
  FQuestion := AQuestion;
  FAnswer := AAnswer;
end;

procedure TThreadedQATask.Exec;
begin
  FState := ChatLLMQARank(FLLM.FObj, PUTF8Char(UTF8Encode(FQuestion)), PUTF8Char(UTF8Encode(FAnswer)));
end;

{ TThreadedToolInputTask }

constructor TThreadedToolInputTask.Create(ALLM: TChatLLM; AInput: string);
begin
  inherited Create(ALLM);
  FInput := AInput;
end;

procedure TThreadedToolInputTask.Exec;
begin
  FState := ChatLLMToolInput(FLLM.FObj, PUTF8Char(UTF8Encode(FInput)));
end;

{ TThreadedChatTask }

constructor TThreadedChatTask.Create(ALLM: TChatLLM; AInput: string);
begin
  inherited Create(ALLM);
  FInput := AInput;
end;

procedure TThreadedChatTask.Exec;
begin
  FState := ChatLLMUserInput(FLLM.FObj, PUTF8Char(UTF8Encode(FInput)));
end;

{ TThreadedTask }

constructor TThreadedTask.Create(ALLM: TChatLLM);
begin
  FLLM := ALLM;
end;

function _RunThreadedTask(Parameter : Pointer): IntPtr;
var
  O: TThreadedTask;
begin
  O := TThreadedTask(Parameter);
  O.Exec;
  TThread.Synchronize(nil, O.SyncedProc);
  O.Free;
  Result := 0;
end;

procedure TThreadedTask.Start(ANext: TLLMAPIEnded);
begin
  FNext := ANext;
{$ifdef dcc}
  var T := TTask.Create(procedure
    begin
      _RunThreadedTask(Self);
    end);
  T.Start;
{$else}
  BeginThread(_RunThreadedTask, Self);
{$endif}
end;

procedure TThreadedTask.SyncedProc;
begin
  FNext(FState);
end;

{ TLLMEnd }

constructor TLLMEnd.Create(ALLM: TChatLLM);
begin
  FLLM := ALLM;
end;

procedure TLLMEnd.SyncedRun;
begin
  FLLM.HandleEnd;
  Free;
end;

{ TLLMPrintItem }

constructor TLLMPrintItem.Create(ALLM: TChatLLM; APrintType: Integer; AUTF8Str: PAnsiChar);
begin
  FLLM := ALLM;
  FPrintType := APrintType;
  FStr := string(UTF8ToString(AUTF8Str));
end;

procedure TLLMPrintItem.SyncedRun;
begin
  FLLM.HandlePrint(FPrintType, FStr);
  Free;
end;

{ TChatLLM }

procedure TChatLLM.AbortGeneration;
begin
  ChatllmAbortGeneration(FObj);
end;

function TChatLLM.CallChat(const AInput: string; OnResult: TLLMPrintEvent
  ): Integer;
begin
  if Busy then Exit(-1);

  FReferences.Clear;
  Result := 0;
  SetBusy(True);

  FCallingMode := True;
  FCallResult := OnResult;
  TThreadedChatTask.Create(Self, AInput).Start(CallChatEnd);
end;

procedure TChatLLM.AddParam(AParams: array of string);
var
  S: string;
begin
  for S in AParams do
    AddParam(S);
end;

procedure TChatLLM.AddParam(AParam: string);
begin
  ChatLLMAppendParam(FObj, PAnsiChar(UTF8Encode(AParam)));
end;

procedure TChatLLM.AddParam(AParams: TStrings);
var
  S: string;
begin
  for S in AParams do
    AddParam(S);
end;

function TChatLLM.Chat(const AInput: string): Integer;
begin
  if Busy then Exit(-1);
  DoBeforeChat;
  FReferences.Clear;
  FOutputAcc := '';
  FThinking := False;
  FThoughtAcc := '';
  Result := 0;
  SetBusy(True);

  TThreadedChatTask.Create(Self, AInput).Start(ChatEnd);
end;

constructor TChatLLM.Create;
begin
  inherited;
  FReferences := TStringList.Create;
  FObj := ChatLLMCreate;
end;

destructor TChatLLM.Destroy;
begin
  FReferences.Free;
  inherited;
end;

procedure TChatLLM.HandleEnd;
begin
  if Assigned(FOnGenerationEnded) then
    FOnGenerationEnded(Self);
end;

procedure TChatLLM.HandlePrint(APrintType: Integer; S: string);
begin
  case APrintType of
    Ord(TPrintType.PRINT_CHAT_CHUNK):
      begin
        FOutputAcc := FOutputAcc + S;
        if Assigned(FOnChunk) then
          FOnChunk(Self, S);
      end;
    Ord(TPrintType.PRINTLN_META):
      begin
        if Assigned(FOnPrintMeta) then
          FOnPrintMeta(Self, S)
        else
          FMetaAcc := FMetaAcc + S;
      end;
    Ord(TPrintType.PRINTLN_ERROR):
      begin
        if Assigned(FOnPrintError) then
          FOnPrintError(Self, S)
        else;
      end;
    Ord(TPrintType.PRINTLN_REF):
      begin
        FReferences.Add(S);
        if Assigned(FOnPrintReference) then
          FOnPrintReference(Self, S);
      end;
    Ord(TPrintType.PRINTLN_REWRITTEN_QUERY):
      begin
        if Assigned(FOnPrintRewrittenQuery) then
          FOnPrintRewrittenQuery(Self, S);
      end;
    Ord(TPrintType.PRINTLN_HISTORY_USER):
      begin
        if Assigned(FOnPrintHistoryUser) then
          FOnPrintHistoryUser(Self, S);
      end;
    Ord(TPrintType.PRINTLN_HISTORY_AI):
      begin
        if Assigned(FOnPrintHistoryAI) then
          FOnPrintHistoryAI(Self, S);
      end;
    Ord(TPrintType.PRINTLN_TOOL_CALLING):
      begin
        if Assigned(FOnPrintToolCalling) then
          FOnPrintToolCalling(Self, S);
      end;
    Ord(TPrintType.PRINTLN_EMBEDDING), Ord(TPrintType.PRINTLN_RANKING):
      FMiscResult := S;
    Ord(TPrintType.PRINTLN_MODEL_INFO):
      FModelInfo := S;
    Ord(TPrintType.PRINT_THOUGHT_CHUNK):
      begin
        FThinking := True;
        FThoughtAcc := FThoughtAcc + S;
        if Assigned(FOnThoughtChunk) then
          FOnThoughtChunk(Self, S);
      end;
    Ord(TPrintType.PRINT_EVT_THOUGHT_COMPLETED):
      begin
        FThinking := False;
        if Assigned(FOnThoughtEnded) then
          FOnThoughtEnded(Self);
      end;
  end;
end;

function TChatLLM.QARanking(const AQustion, AAnswer: string): Integer;
begin
  if Busy then Exit(-1);

  Result := 0;
  SetBusy(True);
  FMiscResult := '';

  TThreadedQATask.Create(Self, AQustion, AAnswer).Start(QARankingEnd);
end;

function TChatLLM.RAGSelectStore(const AName: string): Integer;
begin
  Result := ChatLLMRAGSelectStore(FObj, PAnsiChar(UTF8Encode(AName)));
end;

procedure TChatLLM.Restart(ASysPrompt: string);
begin
  ChatLLMRestart(FObj, PAnsiChar(UTF8Encode(ASysPrompt)));
end;

procedure TChatLLM.Restart;
begin
  ChatLLMRestart(FObj, nil);
end;

procedure TChatLLM.SetAIPrefix(const Value: string);
begin
  ChatLLMSetAIPrefix(FObj, PAnsiChar(UTF8Encode(Value)));
end;

procedure TChatLLM.InternalChunk(S: string);
begin

end;

procedure TChatLLM.DoBeforeChat;
begin

end;

procedure TChatLLM.SetBusy(AValue: Boolean);
var
  F: Boolean;
begin
  F := False;
  if AValue then
  begin
    Inc(FBusy);
    if FBusy = 1 then F := True;
  end else
  begin
    Dec(FBusy);
    if FBusy = 0 then F := True;
  end;

  if F and Assigned(FOnStateChanged) then
    FOnStateChanged(Self, GetBusy);
end;

procedure TChatLLM.SetGenMaxTokens(const Value: Integer);
begin
  ChatLLMSetGenMaxTokens(FObj, Value);
end;

procedure TChatLLM.SetOnChunk(const Value: TLLMPrintEvent);
begin
  FOnChunk := Value;
end;

function TChatLLM.GetBusy: Boolean;
begin
  Result := FBusy > 0;
end;

procedure TChatLLM.SetOnGenerationEnded(const Value: TNotifyEvent);
begin
  FOnGenerationEnded := Value;
end;

procedure TChatLLM.SetOnPrintError(const Value: TLLMPrintEvent);
begin
  FOnPrintError := Value;
end;

procedure TChatLLM.SetOnPrintHistoryAI(const Value: TLLMPrintEvent);
begin
  FOnPrintHistoryAI := Value;
end;

procedure TChatLLM.SetOnPrintHistoryUser(const Value: TLLMPrintEvent);
begin
  FOnPrintHistoryUser := Value;
end;

procedure TChatLLM.SetOnPrintMeta(const Value: TLLMPrintEvent);
begin
  FOnPrintMeta := Value;
end;

procedure TChatLLM.SetOnPrintReference(const Value: TLLMPrintEvent);
begin
  FOnPrintReference := Value;
end;

procedure TChatLLM.SetOnPrintRewrittenQuery(const Value: TLLMPrintEvent);
begin
  FOnPrintRewrittenQuery := Value;
end;

procedure TChatLLM.SetOnPrintToolCalling(const Value: TLLMPrintEvent);
begin
  FOnPrintToolCalling := Value;
end;

procedure TChatLLM.SetOnQARankingResult(const Value: TLLMQARankingResult);
begin
  FOnQARankingResult := Value;
end;

procedure TChatLLM.SetOnStateChanged(const Value: TLLMStateChangedEvent);
begin
  FOnStateChanged := Value;
end;

procedure TChatLLM.SetOnTextEmbeddingResult(
  const Value: TLLMTextEmbeddingResult);
begin
  FOnTextEmbeddingResult := Value;
end;

procedure TChatLLM.ChatEnd(AState: Integer);
begin
  SetBusy(False);
end;

procedure TChatLLM.CallChatEnd(AState: Integer);
begin
  FCallingMode := False;
  FCallResult(Self, FOutputAcc);
  SetBusy(False);
end;

procedure TChatLLM.TextEmbeddingEnd(AState: Integer);
var
  A: array of Single = nil;
{$ifdef dcc}
  L: TArray<string>;
{$else}
  L: array of string;
{$endif}
  I: Integer;
begin
  SetBusy(False);
  if not Assigned(FOnTextEmbeddingResult) then Exit;

  L := FMiscResult.Split([',', ' ', #10, #13], TStringSplitOptions.ExcludeEmpty);
  SetLength(A, Length(L));
  for I := 0 to Length(L) - 1 do
  begin
    A[I] := StrToFloatDef(L[I], 0.0);
  end;

  FOnTextEmbeddingResult(Self, AState, A);
end;

procedure TChatLLM.QARankingEnd(AState: Integer);
begin
  SetBusy(False);
  if not Assigned(FOnQARankingResult) then Exit;
  FOnQARankingResult(Self, AState, StrToFloatDef(FMiscResult, 0.0));
end;

function TChatLLM.Start: Integer;
begin
  Result := ChatLLMStart(FObj, @_LLMPrint, @_LLMEnd, Self);
end;

function TChatLLM.TextEmbedding(const AText: string; const APurpose: TEmbeddingPurpose): Integer;
begin
  if Busy then Exit(-1);

  Result := 0;
  SetBusy(True);
  FMiscResult := '';

  TThreadedEmbeddingTask.Create(Self, AText, APurpose).Start(TextEmbeddingEnd);
end;

function TChatLLM.ToolCompletion(const AInput: string): Integer;
begin
  Result := 0;
  SetBusy(True);

  TThreadedToolCompletionTask.Create(Self, AInput).Start(ChatEnd);
end;

function TChatLLM.ToolInput(const AInput: string): Integer;
begin
  Result := 0;
  SetBusy(True);

  TThreadedToolInputTask.Create(Self, AInput).Start(ChatEnd);
end;

end.

