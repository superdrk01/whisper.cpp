package com.example.whisper;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Thread-safe native whisper.cpp engine. One transcription can run per instance. */
public final class WhisperEngine implements Closeable {
    static { System.loadLibrary("whisper_android"); }

    public static final int STRATEGY_GREEDY = 0;
    public static final int STRATEGY_BEAM_SEARCH = 1;

    public static final int AHEADS_NONE = 0;
    public static final int AHEADS_N_TOP_MOST = 1;
    public static final int AHEADS_CUSTOM = 2;
    public static final int AHEADS_TINY_EN = 3;
    public static final int AHEADS_TINY = 4;
    public static final int AHEADS_BASE_EN = 5;
    public static final int AHEADS_BASE = 6;
    public static final int AHEADS_SMALL_EN = 7;
    public static final int AHEADS_SMALL = 8;
    public static final int AHEADS_MEDIUM_EN = 9;
    public static final int AHEADS_MEDIUM = 10;
    public static final int AHEADS_LARGE_V1 = 11;
    public static final int AHEADS_LARGE_V2 = 12;
    public static final int AHEADS_LARGE_V3 = 13;
    public static final int AHEADS_LARGE_V3_TURBO = 14;

    private volatile long handle;

    private WhisperEngine(long handle) { this.handle = handle; }

    public static WhisperEngine create(ModelOptions options) {
        Objects.requireNonNull(options, "options");
        long h = nativeCreate(options.modelPath, options.useGpu, options.flashAttention,
                options.gpuDevice, options.enableDtw, options.dtwPreset,
                options.dtwTopHeads, options.dtwMemoryBytes);
        if (h == 0) throw new IllegalStateException("Could not load Whisper model");
        return new WhisperEngine(h);
    }

    /** Tries GPU first, then CPU while preserving DTW settings. */
    public static WhisperEngine createWithCpuFallback(ModelOptions options) {
        try {
            return create(options);
        } catch (RuntimeException gpuFailure) {
            if (!options.useGpu) throw gpuFailure;
            return create(options.toBuilder().setUseGpu(false).build());
        }
    }

    public synchronized TranscriptionResult transcribe(float[] monoPcm16k, TranscribeOptions o) {
        ensureOpen();
        Objects.requireNonNull(monoPcm16k, "monoPcm16k");
        Objects.requireNonNull(o, "options");
        String json = nativeTranscribe(handle, monoPcm16k, o.language, o.translate,
                o.strategy, o.threads, o.offsetMs, o.durationMs, o.noContext,
                o.singleSegment, o.tokenTimestamps, o.splitOnWord, o.maxLength,
                o.maxTokens, o.suppressBlank, o.suppressNonSpeechTokens,
                o.temperature, o.temperatureIncrement, o.entropyThreshold,
                o.logProbabilityThreshold, o.noSpeechThreshold, o.maxInitialTimestamp,
                o.lengthPenalty, o.bestOf, o.beamSize,
                o.vadEnabled, o.vadModelPath, o.vadThreshold,
                o.vadMinSpeechDurationMs, o.vadMinSilenceDurationMs,
                o.vadMaxSpeechDurationSeconds, o.vadSpeechPadMs, o.vadSamplesOverlapSeconds);
        if (json == null) throw new IllegalStateException("Native transcription returned null");
        return TranscriptionResult.fromJson(json);
    }

    public void cancel() { long h = handle; if (h != 0) nativeCancel(h); }
    public static String systemInfo() { return nativeSystemInfo(); }

    @Override public synchronized void close() {
        if (handle != 0) { nativeDestroy(handle); handle = 0; }
    }
    private void ensureOpen() { if (handle == 0) throw new IllegalStateException("Engine is closed"); }

    public static final class ModelOptions {
        public final String modelPath;
        public final boolean useGpu, flashAttention, enableDtw;
        public final int gpuDevice, dtwPreset, dtwTopHeads;
        public final long dtwMemoryBytes;
        private ModelOptions(Builder b) {
            modelPath=b.modelPath; useGpu=b.useGpu; flashAttention=b.flashAttention;
            gpuDevice=b.gpuDevice; enableDtw=b.enableDtw; dtwPreset=b.dtwPreset;
            dtwTopHeads=b.dtwTopHeads; dtwMemoryBytes=b.dtwMemoryBytes;
        }
        public Builder toBuilder() { return new Builder(modelPath).setUseGpu(useGpu)
                .setFlashAttention(flashAttention).setGpuDevice(gpuDevice)
                .setDtw(enableDtw, dtwPreset).setDtwTopHeads(dtwTopHeads)
                .setDtwMemoryBytes(dtwMemoryBytes); }
        public static final class Builder {
            private final String modelPath;
            private boolean useGpu=true, flashAttention=false, enableDtw=false;
            private int gpuDevice=0, dtwPreset=AHEADS_NONE, dtwTopHeads=-1;
            private long dtwMemoryBytes=0;
            public Builder(String modelPath) { this.modelPath=Objects.requireNonNull(modelPath); }
            public Builder setUseGpu(boolean v){useGpu=v;return this;}
            public Builder setFlashAttention(boolean v){flashAttention=v;return this;}
            public Builder setGpuDevice(int v){gpuDevice=v;return this;}
            public Builder setDtw(boolean enabled,int preset){enableDtw=enabled;dtwPreset=preset;return this;}
            public Builder setDtwTopHeads(int v){dtwTopHeads=v;return this;}
            public Builder setDtwMemoryBytes(long v){dtwMemoryBytes=v;return this;}
            public ModelOptions build(){return new ModelOptions(this);}
        }
    }

    public static final class TranscribeOptions {
        public final String language;
        public final boolean translate,noContext,singleSegment,tokenTimestamps,splitOnWord,suppressBlank,suppressNonSpeechTokens,vadEnabled;
        public final int strategy,threads,offsetMs,durationMs,maxLength,maxTokens,bestOf,beamSize;
        public final float temperature,temperatureIncrement,entropyThreshold,logProbabilityThreshold,noSpeechThreshold,maxInitialTimestamp,lengthPenalty,vadThreshold,vadMaxSpeechDurationSeconds,vadSamplesOverlapSeconds;
        public final String vadModelPath;
        public final int vadMinSpeechDurationMs,vadMinSilenceDurationMs,vadSpeechPadMs;
        private TranscribeOptions(Builder b){language=b.language;translate=b.translate;strategy=b.strategy;threads=b.threads;
            offsetMs=b.offsetMs;durationMs=b.durationMs;noContext=b.noContext;singleSegment=b.singleSegment;
            tokenTimestamps=b.tokenTimestamps;splitOnWord=b.splitOnWord;maxLength=b.maxLength;maxTokens=b.maxTokens;
            suppressBlank=b.suppressBlank;suppressNonSpeechTokens=b.suppressNonSpeechTokens;temperature=b.temperature;
            temperatureIncrement=b.temperatureIncrement;entropyThreshold=b.entropyThreshold;logProbabilityThreshold=b.logProbabilityThreshold;
            noSpeechThreshold=b.noSpeechThreshold;maxInitialTimestamp=b.maxInitialTimestamp;lengthPenalty=b.lengthPenalty;
            bestOf=b.bestOf;beamSize=b.beamSize;vadEnabled=b.vadEnabled;vadModelPath=b.vadModelPath;
            vadThreshold=b.vadThreshold;vadMinSpeechDurationMs=b.vadMinSpeechDurationMs;
            vadMinSilenceDurationMs=b.vadMinSilenceDurationMs;vadMaxSpeechDurationSeconds=b.vadMaxSpeechDurationSeconds;
            vadSpeechPadMs=b.vadSpeechPadMs;vadSamplesOverlapSeconds=b.vadSamplesOverlapSeconds;}
        public static final class Builder {
            private String language=null,vadModelPath=null; private boolean translate=false,noContext=false,singleSegment=false,tokenTimestamps=true,splitOnWord=true,suppressBlank=true,suppressNonSpeechTokens=false,vadEnabled=false;
            private int strategy=STRATEGY_GREEDY,threads=Math.max(1,Math.min(4,Runtime.getRuntime().availableProcessors())),offsetMs=0,durationMs=0,maxLength=42,maxTokens=0,bestOf=5,beamSize=5;
            private float temperature=0f,temperatureIncrement=0.2f,entropyThreshold=2.4f,logProbabilityThreshold=-1f,noSpeechThreshold=0.6f,maxInitialTimestamp=1f,lengthPenalty=-1f;
            private float vadThreshold=0.5f,vadMaxSpeechDurationSeconds=Float.MAX_VALUE,vadSamplesOverlapSeconds=0.10f;
            private int vadMinSpeechDurationMs=250,vadMinSilenceDurationMs=100,vadSpeechPadMs=30;
            public Builder setLanguage(String v){language=v;return this;} public Builder setTranslate(boolean v){translate=v;return this;}
            public Builder setStrategy(int v){strategy=v;return this;} public Builder setThreads(int v){threads=v;return this;}
            public Builder setOffsetMs(int v){offsetMs=v;return this;} public Builder setDurationMs(int v){durationMs=v;return this;}
            public Builder setNoContext(boolean v){noContext=v;return this;} public Builder setSingleSegment(boolean v){singleSegment=v;return this;}
            public Builder setTokenTimestamps(boolean v){tokenTimestamps=v;return this;} public Builder setSplitOnWord(boolean v){splitOnWord=v;return this;}
            public Builder setMaxLength(int v){maxLength=v;return this;} public Builder setMaxTokens(int v){maxTokens=v;return this;}
            public Builder setSuppressBlank(boolean v){suppressBlank=v;return this;} public Builder setSuppressNonSpeechTokens(boolean v){suppressNonSpeechTokens=v;return this;}
            public Builder setTemperature(float v){temperature=v;return this;} public Builder setTemperatureIncrement(float v){temperatureIncrement=v;return this;}
            public Builder setEntropyThreshold(float v){entropyThreshold=v;return this;} public Builder setLogProbabilityThreshold(float v){logProbabilityThreshold=v;return this;}
            public Builder setNoSpeechThreshold(float v){noSpeechThreshold=v;return this;} public Builder setMaxInitialTimestamp(float v){maxInitialTimestamp=v;return this;}
            public Builder setLengthPenalty(float v){lengthPenalty=v;return this;} public Builder setBestOf(int v){bestOf=v;return this;}
            public Builder setBeamSize(int v){beamSize=v;return this;}
            /** Enables whisper.cpp's integrated Silero VAD. The model must be a ggml Silero VAD .bin file. */
            public Builder setVad(boolean enabled,String modelPath){vadEnabled=enabled;vadModelPath=modelPath;return this;}
            public Builder setVadThreshold(float v){vadThreshold=v;return this;}
            public Builder setVadMinSpeechDurationMs(int v){vadMinSpeechDurationMs=v;return this;}
            public Builder setVadMinSilenceDurationMs(int v){vadMinSilenceDurationMs=v;return this;}
            public Builder setVadMaxSpeechDurationSeconds(float v){vadMaxSpeechDurationSeconds=v;return this;}
            public Builder setVadSpeechPadMs(int v){vadSpeechPadMs=v;return this;}
            public Builder setVadSamplesOverlapSeconds(float v){vadSamplesOverlapSeconds=v;return this;}
            public TranscribeOptions build(){
                if(vadEnabled && (vadModelPath==null || vadModelPath.isEmpty())) throw new IllegalArgumentException("VAD model path is required when VAD is enabled");
                if(vadThreshold<0f || vadThreshold>1f) throw new IllegalArgumentException("VAD threshold must be between 0 and 1");
                if(vadMinSpeechDurationMs<0 || vadMinSilenceDurationMs<0 || vadSpeechPadMs<0) throw new IllegalArgumentException("VAD durations cannot be negative");
                if(vadMaxSpeechDurationSeconds<=0f || vadSamplesOverlapSeconds<0f) throw new IllegalArgumentException("Invalid VAD duration/overlap");
                return new TranscribeOptions(this);
            }
        }
    }

    public static final class Token {
        public final int id; public final String text; public final long startMs,endMs,dtwMs; public final double probability;
        Token(JSONObject j)throws JSONException{id=j.getInt("id");text=j.getString("text");startMs=j.getLong("startMs");endMs=j.getLong("endMs");dtwMs=j.getLong("dtwMs");probability=j.getDouble("probability");}
    }
    public static final class Segment {
        public final long startMs,endMs; public final String text; public final List<Token> tokens;
        Segment(JSONObject j)throws JSONException{startMs=j.getLong("startMs");endMs=j.getLong("endMs");text=j.getString("text");JSONArray a=j.getJSONArray("tokens");List<Token>x=new ArrayList<>();for(int i=0;i<a.length();i++)x.add(new Token(a.getJSONObject(i)));tokens=Collections.unmodifiableList(x);}
    }
    public static final class VadSegment {
        public final long startMs,endMs;
        VadSegment(JSONObject j)throws JSONException{startMs=j.getLong("startMs");endMs=j.getLong("endMs");}
    }
    public static final class TranscriptionResult {
        public final int status,detectedLanguageId; public final boolean cancelled,vadEnabled;
        public final List<Segment> segments; public final List<VadSegment> vadSegments;
        private TranscriptionResult(int s,int l,boolean c,boolean v,List<Segment>x,List<VadSegment>vs){status=s;detectedLanguageId=l;cancelled=c;vadEnabled=v;segments=Collections.unmodifiableList(x);vadSegments=Collections.unmodifiableList(vs);}
        static TranscriptionResult fromJson(String raw){try{JSONObject j=new JSONObject(raw);JSONArray a=j.getJSONArray("segments");List<Segment>x=new ArrayList<>();for(int i=0;i<a.length();i++)x.add(new Segment(a.getJSONObject(i)));JSONArray va=j.optJSONArray("vadSegments");List<VadSegment>vs=new ArrayList<>();if(va!=null)for(int i=0;i<va.length();i++)vs.add(new VadSegment(va.getJSONObject(i)));return new TranscriptionResult(j.getInt("status"),j.optInt("detectedLanguageId",-1),j.optBoolean("cancelled"),j.optBoolean("vadEnabled"),x,vs);}catch(JSONException e){throw new IllegalStateException("Invalid native result",e);}}
        public boolean isSuccessful(){return status==0;}
    }

    private static native long nativeCreate(String modelPath, boolean useGpu, boolean flashAttention, int gpuDevice, boolean enableDtw, int dtwPreset, int dtwTopHeads, long dtwMemoryBytes);
    private static native void nativeDestroy(long handle);
    private static native void nativeCancel(long handle);
    private static native String nativeSystemInfo();
    private static native String nativeTranscribe(long handle,float[] audio,String language,boolean translate,int strategy,int threads,int offsetMs,int durationMs,boolean noContext,boolean singleSegment,boolean tokenTimestamps,boolean splitOnWord,int maxLength,int maxTokens,boolean suppressBlank,boolean suppressNst,float temperature,float temperatureIncrement,float entropyThreshold,float logProbabilityThreshold,float noSpeechThreshold,float maxInitialTimestamp,float lengthPenalty,int bestOf,int beamSize,boolean vadEnabled,String vadModelPath,float vadThreshold,int vadMinSpeechDurationMs,int vadMinSilenceDurationMs,float vadMaxSpeechDurationSeconds,int vadSpeechPadMs,float vadSamplesOverlapSeconds);
}
