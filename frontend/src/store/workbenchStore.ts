import { create, StateCreator } from "zustand";

type BacktestWindow = "test" | "train" | "all";

interface WorkbenchState {
  // View / filter
  backtestWindow: BacktestWindow;
  filterText: string;
  sourceFilter: string;
  neutralize: boolean;

  // Mining params
  mPopSize: number;
  mMaxK: number;
  mFolds: number;
  mEmbargo: number;
  mPurge: number;
  mLambdaStd: number;
  mLambdaCx: number;
  mLambdaSize: number;
  mSizeCorr: number;

  // Backtest job state
  jobId: string | null;
  isLaunching: boolean;
  launchError: string | null;

  // Mining job state
  miningJobId: string | null;
  miningLaunching: boolean;
  miningError: string | null;

  // Actions
  setBacktestWindow: (v: BacktestWindow) => void;
  setFilterText: (v: string) => void;
  setSourceFilter: (v: string) => void;
  setNeutralize: (v: boolean) => void;
  setMPopSize: (v: number) => void;
  setMMaxK: (v: number) => void;
  setMFolds: (v: number) => void;
  setMEmbargo: (v: number) => void;
  setMPurge: (v: number) => void;
  setMLambdaStd: (v: number) => void;
  setMLambdaCx: (v: number) => void;
  setMLambdaSize: (v: number) => void;
  setMSizeCorr: (v: number) => void;
  setJobId: (v: string | null) => void;
  setIsLaunching: (v: boolean) => void;
  setLaunchError: (v: string | null) => void;
  setMiningJobId: (v: string | null) => void;
  setMiningLaunching: (v: boolean) => void;
  setMiningError: (v: string | null) => void;
}

const storeCreator: StateCreator<WorkbenchState> = (set) => ({
  // defaults
  backtestWindow: "test",
  filterText: "",
  sourceFilter: "EVO",
  neutralize: true,
  mPopSize: 300,
  mMaxK: 15,
  mFolds: 5,
  mEmbargo: 5,
  mPurge: 10,
  mLambdaStd: 0.50,
  mLambdaCx: 0.001,
  mLambdaSize: 0.50,
  mSizeCorr: 0.70,
  jobId: null,
  isLaunching: false,
  launchError: null,
  miningJobId: null,
  miningLaunching: false,
  miningError: null,

  // actions
  setBacktestWindow: (v) => set({ backtestWindow: v }),
  setFilterText: (v) => set({ filterText: v }),
  setSourceFilter: (v) => set({ sourceFilter: v }),
  setNeutralize: (v) => set({ neutralize: v }),
  setMPopSize: (v) => set({ mPopSize: v }),
  setMMaxK: (v) => set({ mMaxK: v }),
  setMFolds: (v) => set({ mFolds: v }),
  setMEmbargo: (v) => set({ mEmbargo: v }),
  setMPurge: (v) => set({ mPurge: v }),
  setMLambdaStd: (v) => set({ mLambdaStd: v }),
  setMLambdaCx: (v) => set({ mLambdaCx: v }),
  setMLambdaSize: (v) => set({ mLambdaSize: v }),
  setMSizeCorr: (v) => set({ mSizeCorr: v }),
  setJobId: (v) => set({ jobId: v }),
  setIsLaunching: (v) => set({ isLaunching: v }),
  setLaunchError: (v) => set({ launchError: v }),
  setMiningJobId: (v) => set({ miningJobId: v }),
  setMiningLaunching: (v) => set({ miningLaunching: v }),
  setMiningError: (v) => set({ miningError: v }),
});

export const useWorkbenchStore = create<WorkbenchState>(storeCreator);
