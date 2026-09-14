"""One public CPU/library contract shared by prompts and candidate workers."""
PORTFOLIO_LIBRARIES=('numpy','scipy','pandas','sklearn','statsmodels','sympy','matplotlib','xgboost','cvxpy')
COMPUTATIONAL_STDLIB=('math','cmath','statistics','random','collections','itertools','functools','operator',
    'typing','dataclasses','enum','decimal','fractions','heapq','bisect','array','copy','time','numbers','warnings')
CVXPY_SOLVERS=('CLARABEL','OSQP','SCS','SCIPY','HIGHS')
PORTFOLIO_CPUS=4
PORTFOLIO_MEMORY_GIB=8


def library_environment(cpus):
    if type(cpus) is not int or cpus<1:raise ValueError('CPU allowance must be positive')
    # One native thread per joblib task avoids N workers times N BLAS threads.
    # The task can deliberately use a single multithreaded algorithm instead,
    # while the outer CPU allocation still bounds all descendants together.
    return dict(OPENBLAS_NUM_THREADS='1',OPENBLAS_DEFAULT_NUM_THREADS='1',
        GOTO_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
        OMP_NUM_THREADS='1',OMP_THREAD_LIMIT=str(cpus),OMP_MAX_ACTIVE_LEVELS='1',OMP_DYNAMIC='FALSE',
        MKL_DYNAMIC='FALSE',NUMEXPR_NUM_THREADS='1',NUMEXPR_MAX_THREADS=str(cpus),
        LOKY_MAX_CPU_COUNT=str(cpus),RAYON_NUM_THREADS=str(cpus),
        SCIENCE_CPUS=str(cpus),MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/matplotlib',
        JOBLIB_TEMP_FOLDER='/tmp/joblib')


def portfolio_prompt_resources(cpus=PORTFOLIO_CPUS,memory_gib=PORTFOLIO_MEMORY_GIB):
    return ('Allowed third-party imports: '+', '.join(PORTFOLIO_LIBRARIES)+'. '
        'scikit-learn is imported as sklearn. statsmodels.api and statsmodels.tsa are allowed. '
        'matplotlib uses the noninteractive Agg backend; plots do not affect reward.\n'
        f'XGBoost: CPU-only build; use device="cpu", tree_method="hist" and n_jobs/nthread <= {cpus}. '
        'CVXPY: installed solver allowlist is '+', '.join(CVXPY_SOLVERS)+'. '
        'Set solver-specific time/iteration/thread options within the task budget when supported; '
        'there are no commercial, remote, or GPU solvers.\n'
        'Allowed standard-library imports: '+', '.join(COMPUTATIONAL_STDLIB)+'.\n'
        f'Resources: {cpus} CPU cores and {memory_gib} GiB shared by the complete candidate process tree. '
        'No accelerator. fit: 180 seconds including imports and training; inference/replay: 60 seconds; '
        'overall: 300 seconds. Serialized fitted model limit: 128 MiB.\n'
        f'Parallelism: use n_jobs between 1 and {cpus} (or -1 to use the assigned CPUs). '
        'The harness defaults joblib to threading and native BLAS/OpenMP/NumExpr to one thread per worker, '
        'avoiding nested oversubscription. NumPy, SciPy, statsmodels and scikit-learn share these numerical '
        'thread pools; pandas may also use NumExpr. SymPy and matplotlib have no universal per-library '
        'memory/thread quota. All libraries share the same CPU, memory and time budget; '
        'they do not each receive another allocation. Do not override resource controls.\n'
        'JAX, Flax, Optax, PyTorch, network/filesystem access, downloads, external checkpoints and '
        'user-launched external processes are prohibited. Library-managed computational parallelism '
        'within the allocation is permitted. Use the supplied numpy.random.Generator.')
