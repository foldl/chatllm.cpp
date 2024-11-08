program main;

{$APPTYPE CONSOLE}

{$ifdef fpc}{$mode delphi}{$endif}

uses
  Math, LibChatLLM;

procedure _LLMPrint(UserData: Pointer; APrintType: Integer; AUTF8Str: PAnsiChar); cdecl;
var
  T: TPrintType;
begin
  T := TPrintType(APrintType);
  case T of
    PRINT_CHAT_CHUNK: Write(UTF8ToString(AUTF8Str));
  else
    WriteLn(UTF8ToString(AUTF8Str));
  end;

end;

procedure _LLMEnd(UserData: Pointer); cdecl; begin end;

var
  LLM: PChatLLMObj;
  Input: string;
  I: Integer;

begin
  SetExceptionMask([Low(TFPUException)..High(TFPUException)]);

  LLM := ChatLLMCreate;

  for I := 1 to ParamCount do
    ChatLLMAppendParam(LLM, PUTF8Char(UTF8Encode(ParamStr(I))));

  if ChatLLMStart(LLM, @_LLMPrint, @_LLMEnd, nil) <> 0 then Exit;

  while True do
  begin
    Write('You  > ');
    ReadLn(Input);
    if Length(Input) < 1 then Continue;
    Write('A.I. > ');
    ChatLLMUserInput(LLM, PUTF8Char(UTF8Encode(Input)));
    WriteLn;
  end;
end.