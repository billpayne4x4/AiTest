namespace AiModel_V1.Gpu;

/// <summary>
/// Embedded PTX kernels for the zero-dependency GPU backend. PTX is a virtual
/// ISA that the NVIDIA driver JIT-compiles, so this single source runs on any
/// CUDA-capable GPU (Maxwell → Blackwell) on Windows and Linux.
///
/// Two kernels mirror <see cref="NeuralNetwork"/> exactly (same constants,
/// same update order, same simplifications such as pass-through LayerNorm
/// backward). Conventions:
///   - one thread block per launch, blockDim = max layer width (≤ 1024)
///   - all float buffers are f32; tables are u32 element indices
///   - flags bit0 = LayerNorm, bit1 = residual, bit2 = output layer
///   - act: 0 = tanh, 1 = relu, 2 = gelu
///   - calls are never predicated; only loads/stores/branches use guards,
///     and out-of-range threads reuse index 0 for loads
/// </summary>
public static class PtxKernels
{
    public const string Source = @"
.version 7.0
.target sm_70
.address_size 64

// dynamic shared memory (sized at launch: (blockDim + 2) floats)
.extern .shared .b8 shraw[];

.func (.param .f32 r_ret) tanh_apx(.param .f32 x_par)
{
    .reg .f32 %r, %x, %a, %e, %n, %d, %t;
    .reg .pred %p, %q;
    ld.param.f32 %x, [x_par];
    abs.f32 %a, %x;
    setp.gt.f32 %p, %a, 9.0;
    @%p bra CLAMP_T;
    mul.f32 %t, %x, 2.88539008178;
    ex2.approx.f32 %e, %t;
    sub.f32 %n, %e, 1.0;
    add.f32 %d, %e, 1.0;
    div.full.f32 %t, %n, %d;
    mov.f32 %r, %t;
    bra TANH_RET;
CLAMP_T:
    setp.gt.f32 %q, %x, 0.0;
    selp.f32 %r, 1.0, -1.0, %q;
TANH_RET:
    st.param.f32 [r_ret], %r;
    ret;
}

.func (.param .f32 r_ret) activ_apx(.param .f32 v_par, .param .u32 aid_par)
{
    .reg .f32 %r, %v, %t, %x3, %u;
    .reg .u32 %aid;
    .reg .pred %p1, %p2;
    ld.param.f32 %v, [v_par];
    ld.param.u32 %aid, [aid_par];
    setp.eq.u32 %p1, %aid, 1;
    setp.eq.u32 %p2, %aid, 2;
    @%p1 bra DO_RELU;
    @%p2 bra DO_GELU;
    call (%t), tanh_apx, (%v);
    mov.f32 %r, %t;
    bra ACTIV_RET;
DO_RELU:
    max.f32 %r, %v, 0.0;
    bra ACTIV_RET;
DO_GELU:
    mul.f32 %x3, %v, %v;
    mul.f32 %x3, %x3, %v;
    fma.rn.f32 %u, %x3, 0.044715, %v;
    mul.f32 %u, %u, 0.79788458;
    call (%t), tanh_apx, (%u);
    add.f32 %t, %t, 1.0;
    mul.f32 %t, %t, %v;
    mul.f32 %r, %t, 0.5;
ACTIV_RET:
    st.param.f32 [r_ret], %r;
    ret;
}

.func (.param .f32 r_ret) deriv_apx(.param .f32 z_par, .param .f32 a_par, .param .u32 aid_par)
{
    .reg .f32 %r, %z, %a, %t, %x3, %u;
    .reg .u32 %aid;
    .reg .pred %p1, %p2, %p3;
    ld.param.f32 %z, [z_par];
    ld.param.f32 %a, [a_par];
    ld.param.u32 %aid, [aid_par];
    setp.eq.u32 %p1, %aid, 1;
    setp.eq.u32 %p2, %aid, 2;
    @%p1 bra D_RELU;
    @%p2 bra D_GELU;
    mul.f32 %t, %a, %a;
    sub.f32 %r, 1.0, %t;
    bra DERIV_RET;
D_RELU:
    setp.gt.f32 %p3, %z, 0.0;
    selp.f32 %r, 1.0, 0.0, %p3;
    bra DERIV_RET;
D_GELU:
    mul.f32 %x3, %z, %z;
    mul.f32 %x3, %x3, %z;
    fma.rn.f32 %u, %x3, 0.044715, %z;
    mul.f32 %u, %u, 0.79788458;
    call (%t), tanh_apx, (%u);
    add.f32 %t, %t, 1.0;
    mul.f32 %r, %t, 0.5;
DERIV_RET:
    st.param.f32 [r_ret], %r;
    ret;
}

// ============================================================================
// fwd: whole-network forward pass, layers 0..nl-1 in one block.
// ============================================================================
.visible .entry fwd(
    .param .u64 pW, .param .u64 pB, .param .u64 pA, .param .u64 pN,
    .param .u64 pG, .param .u64 pBt,
    .param .u64 pOffW, .param .u64 pOffB, .param .u64 pOffA,
    .param .u64 pFI, .param .u64 pFO, .param .u64 pFL,
    .param .u32 u_nl, .param .u32 u_aid, .param .f32 f_temp)
{
    .reg .u32 %tx, %l, %nl, %aid, %fi, %fo, %fl, %i, %r1, %rW, %rB, %rAI, %rAO, %idx;
    .reg .u64 %pW, %pB, %pA, %pN, %pG, %pBt, %pOW, %pOB, %pOA, %pFI, %pFO, %pFL;
    .reg .u64 %t1, %t2, %t3, %shbase;
    .reg .f32 %z, %v, %m, %vv, %dd, %iv, %fW, %fA, %fG, %fBe, %fR, %f1, %f2, %a, %temp;
    .reg .pred %p, %ploop, %active, %is0, %isout, %useLn, %useRes;

    mov.u32 %tx, %tid.x;
    ld.param.u64 %pW, [pW];
    ld.param.u64 %pB, [pB];
    ld.param.u64 %pA, [pA];
    ld.param.u64 %pN, [pN];
    ld.param.u64 %pG, [pG];
    ld.param.u64 %pBt, [pBt];
    ld.param.u64 %pOW, [pOffW];
    ld.param.u64 %pOB, [pOffB];
    ld.param.u64 %pOA, [pOffA];
    ld.param.u64 %pFI, [pFI];
    ld.param.u64 %pFO, [pFO];
    ld.param.u64 %pFL, [pFL];
    ld.param.u32 %nl, [u_nl];
    ld.param.u32 %aid, [u_aid];
    ld.param.f32 %temp, [f_temp];
    cvta.shared.u64 %shbase, shraw;

    mov.u32 %l, 0;
FLAYER:
    setp.ge.u32 %p, %l, %nl;
    @%p bra FWD_DONE;
    cvt.u64.u32 %t1, %l;
    shl.b64 %t1, %t1, 2;
    add.u64 %t2, %pFI, %t1;
    ld.global.u32 %fi, [%t2];
    add.u64 %t2, %pFO, %t1;
    ld.global.u32 %fo, [%t2];
    add.u64 %t2, %pFL, %t1;
    ld.global.u32 %fl, [%t2];
    and.b32 %r1, %fl, 1;
    setp.ne.u32 %useLn, %r1, 0;
    and.b32 %r1, %fl, 2;
    setp.ne.u32 %useRes, %r1, 0;
    and.b32 %r1, %fl, 4;
    setp.ne.u32 %isout, %r1, 0;
    setp.lt.u32 %active, %tx, %fo;
    // offsets for this layer
    add.u64 %t2, %pOW, %t1;
    ld.global.u32 %rW, [%t2];
    add.u64 %t2, %pOB, %t1;
    ld.global.u32 %rB, [%t2];
    add.u64 %t2, %pOA, %t1;
    ld.global.u32 %rAI, [%t2];
    add.u32 %r1, %l, 1;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %pOA, %t2;
    ld.global.u32 %rAO, [%t2];
    // safe load index (0 for idle threads)
    selp.u32 %idx, %tx, 0, %active;

    // ---- phase A: z = b + W*a ----
    mov.f32 %z, 0.0;
    @!%active bra A_STORED;
    add.u32 %r1, %rB, %tx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pB, %t3;
    ld.global.f32 %z, [%t3];
    mov.u32 %i, 0;
ACC_LP:
    setp.ge.u32 %ploop, %i, %fi;
    @%ploop bra ACC_DONE;
    mad.lo.u32 %r1, %tx, %fi, %i;
    add.u32 %r1, %r1, %rW;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pW, %t3;
    ld.global.f32 %fW, [%t3];
    add.u32 %r1, %rAI, %i;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pA, %t3;
    ld.global.f32 %fA, [%t3];
    fma.rn.f32 %z, %fW, %fA, %z;
    add.u32 %i, %i, 1;
    bra ACC_LP;
ACC_DONE:
A_STORED:
    mad.lo.u32 %r1, %tx, 4, 8;
    cvt.u64.u32 %t3, %r1;
    add.u64 %t3, %shbase, %t3;
    @%active st.shared.f32 [%t3], %z;
    @!%active st.shared.f32 [%t3], 0.0;
    bar.sync 0;
    // ---- mean/var over fo entries (thread 0) ----
    setp.eq.u32 %is0, %tx, 0;
    @!%is0 bra MV_DONE;
    mov.f32 %m, 0.0;
    mov.u32 %i, 0;
MEAN_LP:
    setp.ge.u32 %ploop, %i, %fo;
    @%ploop bra MEAN_DONE;
    mad.lo.u32 %r1, %i, 4, 8;
    cvt.u64.u32 %t3, %r1;
    add.u64 %t3, %shbase, %t3;
    ld.shared.f32 %f1, [%t3];
    add.f32 %m, %m, %f1;
    add.u32 %i, %i, 1;
    bra MEAN_LP;
MEAN_DONE:
    cvt.rn.f32.u32 %f1, %fo;
    div.full.f32 %m, %m, %f1;
    mov.f32 %vv, 0.0;
    mov.u32 %i, 0;
VAR_LP:
    setp.ge.u32 %ploop, %i, %fo;
    @%ploop bra VAR_DONE;
    mad.lo.u32 %r1, %i, 4, 8;
    cvt.u64.u32 %t3, %r1;
    add.u64 %t3, %shbase, %t3;
    ld.shared.f32 %f1, [%t3];
    sub.f32 %f1, %f1, %m;
    fma.rn.f32 %vv, %f1, %f1, %vv;
    add.u32 %i, %i, 1;
    bra VAR_LP;
VAR_DONE:
    cvt.rn.f32.u32 %f1, %fo;
    div.full.f32 %vv, %vv, %f1;
    add.f32 %vv, %vv, 0.00001;
    rsqrt.approx.f32 %iv, %vv;
    st.shared.f32 [%shbase], %m;
    add.u64 %t3, %shbase, 4;
    st.shared.f32 [%t3], %iv;
MV_DONE:
    bar.sync 0;
    ld.shared.f32 %m, [%shbase];
    add.u64 %t3, %shbase, 4;
    ld.shared.f32 %iv, [%t3];
    // ---- phase B (all threads compute with safe idx; only active store) ----
    add.u32 %r1, %rAO, %idx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pG, %t3;
    ld.global.f32 %fG, [%t3];
    add.u32 %r1, %rAO, %idx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pBt, %t3;
    ld.global.f32 %fBe, [%t3];
    // reload own z for active threads
    mad.lo.u32 %r1, %tx, 4, 8;
    cvt.u64.u32 %t3, %r1;
    add.u64 %t3, %shbase, %t3;
    ld.shared.f32 %z, [%t3];
    sub.f32 %f1, %z, %m;
    mul.f32 %f1, %f1, %iv;
    mul.f32 %f1, %f1, %fG;
    add.f32 %f1, %f1, %fBe;
    selp.f32 %v, %f1, %z, %useLn;
    // store normed
    add.u32 %r1, %rAO, %idx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pN, %t3;
    @%active st.global.f32 [%t3], %v;
    div.full.f32 %f2, %v, %temp;
    selp.f32 %v, %f2, %v, %isout;
    // output layer always uses tanh (aid 0), mirroring CPU Activate(z, isOutput:true)
    selp.u32 %r1, 0, %aid, %isout;
    call (%a), activ_apx, (%v, %r1);
    // residual: a += A_in[tid]
    add.u32 %r1, %rAI, %idx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pA, %t3;
    ld.global.f32 %fR, [%t3];
    add.f32 %f1, %a, %fR;
    selp.f32 %a, %f1, %a, %useRes;
    // clamp hidden to [-6,6]
    min.f32 %f2, %a, 6.0;
    max.f32 %f2, %f2, -6.0;
    selp.f32 %a, %a, %f2, %isout;
    add.u32 %r1, %rAO, %idx;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pA, %t3;
    @%active st.global.f32 [%t3], %a;
    bar.sync 0;
    add.u32 %l, %l, 1;
    bra FLAYER;
FWD_DONE:
    ret;
}

// ============================================================================
// bwd: whole-network reverse update, layers nl-1..0 in one block.
// D blob stride = maxW floats per layer (host-zeroed before launch).
// DN = output-layer dNext (host-formed, deriv already applied).
// ============================================================================
.visible .entry bwd(
    .param .u64 pW, .param .u64 pMW, .param .u64 pVW,
    .param .u64 pMB, .param .u64 pVB, .param .u64 pBB,
    .param .u64 pA, .param .u64 pN, .param .u64 pD, .param .u64 pDN,
    .param .u64 pOffW, .param .u64 pOffB, .param .u64 pOffA,
    .param .u64 pFI, .param .u64 pFO, .param .u64 pFL,
    .param .u32 u_nl, .param .u32 u_aid, .param .u32 u_maxW, .param .u32 u_adam,
    .param .f32 f_lr, .param .f32 f_bc1, .param .f32 f_bc2,
    .param .f32 f_decay, .param .f32 f_bdec, .param .f32 f_maxn)
{
    .reg .u32 %tx, %bd, %l, %nl, %aid, %fi, %fo, %fl, %maxW, %adam;
    .reg .u32 %i, %t, %r1, %rW, %rB, %rAI, %rAO, %islast;
    .reg .u64 %pW, %pMW, %pVW, %pMB, %pVB, %pBB, %pA, %pN, %pD, %pDN;
    .reg .u64 %pOW, %pOB, %pOA, %pFI, %pFO, %pFL;
    .reg .u64 %t1, %t2, %t3, %bDsrc, %bDdst;
    .reg .f32 %dj, %wOld, %g, %m, %vv, %mh, %vh, %dW, %lr, %bc1, %bc2;
    .reg .f32 %dec, %bdec, %maxn, %n2, %nn, %sc, %d, %dd, %zz, %aa, %f1, %f2;
    .reg .pred %p, %ploop, %active, %islastP, %useRes, %useAdam, %pzero, %over;

    mov.u32 %tx, %tid.x;
    mov.u32 %bd, %ntid.x;
    ld.param.u64 %pW, [pW];
    ld.param.u64 %pMW, [pMW];
    ld.param.u64 %pVW, [pVW];
    ld.param.u64 %pMB, [pMB];
    ld.param.u64 %pVB, [pVB];
    ld.param.u64 %pBB, [pBB];
    ld.param.u64 %pA, [pA];
    ld.param.u64 %pN, [pN];
    ld.param.u64 %pD, [pD];
    ld.param.u64 %pDN, [pDN];
    ld.param.u64 %pOW, [pOffW];
    ld.param.u64 %pOB, [pOffB];
    ld.param.u64 %pOA, [pOffA];
    ld.param.u64 %pFI, [pFI];
    ld.param.u64 %pFO, [pFO];
    ld.param.u64 %pFL, [pFL];
    ld.param.u32 %nl, [u_nl];
    ld.param.u32 %aid, [u_aid];
    ld.param.u32 %maxW, [u_maxW];
    ld.param.u32 %adam, [u_adam];
    ld.param.f32 %lr, [f_lr];
    ld.param.f32 %bc1, [f_bc1];
    ld.param.f32 %bc2, [f_bc2];
    ld.param.f32 %dec, [f_decay];
    ld.param.f32 %bdec, [f_bdec];
    ld.param.f32 %maxn, [f_maxn];
    setp.ne.u32 %useAdam, %adam, 0;

    sub.u32 %l, %nl, 1;
BLAYER:
    cvt.u64.u32 %t1, %l;
    shl.b64 %t1, %t1, 2;
    add.u64 %t2, %pFI, %t1;
    ld.global.u32 %fi, [%t2];
    add.u64 %t2, %pFO, %t1;
    ld.global.u32 %fo, [%t2];
    add.u64 %t2, %pFL, %t1;
    ld.global.u32 %fl, [%t2];
    and.b32 %r1, %fl, 2;
    setp.ne.u32 %useRes, %r1, 0;
    setp.lt.u32 %active, %tx, %fo;
    add.u64 %t2, %pOW, %t1;
    ld.global.u32 %rW, [%t2];
    add.u64 %t2, %pOB, %t1;
    ld.global.u32 %rB, [%t2];
    add.u64 %t2, %pOA, %t1;
    ld.global.u32 %rAI, [%t2];
    // dst base = pD + l*maxW (floats->bytes)
    mul.lo.u32 %r1, %l, %maxW;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %bDdst, %pD, %t2;
    // src base: last layer -> DN else D+(l+1)*maxW
    sub.u32 %r1, %nl, 1;
    setp.eq.u32 %islastP, %l, %r1;
    add.u32 %r1, %l, 1;
    mul.lo.u32 %r1, %r1, %maxW;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %pD, %t2;
    selp.u64 %bDsrc, %pDN, %t2, %islastP;

    // ---- phase 0: residual skip (delta was zeroed by host).
    // NOTE: skipped when dNext is exactly 0, mirroring the CPU continue.
    @!%active bra PH0_DONE;
    @!%useRes bra PH0_DONE;
    cvt.u64.u32 %t3, %tx;
    shl.b64 %t3, %t3, 2;
    add.u64 %t2, %bDsrc, %t3;
    ld.global.f32 %dj, [%t2];
    setp.eq.f32 %p, %dj, 0.0;
    @%p bra PH0_DONE;
    add.u64 %t3, %bDdst, %t3;
    st.global.f32 [%t3], %dj;
PH0_DONE:
    // ---- phase 1: row update (thread j = tid) ----
    @!%active bra PH1_DONE;
    cvt.u64.u32 %t3, %tx;
    shl.b64 %t3, %t3, 2;
    add.u64 %t2, %bDsrc, %t3;
    ld.global.f32 %dj, [%t2];
    setp.eq.f32 %pzero, %dj, 0.0;
    @%pzero bra ROW_MAXN;
    mov.u32 %i, 0;
ROW_LP:
    setp.ge.u32 %ploop, %i, %fi;
    @%ploop bra ROW_DONE;
    // wOld = W[rW + tid*fi + i]
    mad.lo.u32 %r1, %tx, %fi, %i;
    add.u32 %r1, %r1, %rW;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %pW, %t3;
    ld.global.f32 %wOld, [%t3];
    // a = A[rAI + i]
    add.u32 %r1, %rAI, %i;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %pA, %t2;
    ld.global.f32 %aa, [%t2];
    // g = dj*a + decay*wOld
    mul.f32 %g, %dj, %aa;
    fma.rn.f32 %g, %dec, %wOld, %g;
    // g is ready in %g; W addr in %t3; wOld loaded
    @%useAdam bra ROW_ADAM;
    // SGD path: dW = lr * g
    mul.f32 %dW, %lr, %g;
    bra ROW_APPLY;
ROW_ADAM:
    // MW/VW share the W layout: byte offset = (rW + tid*fi + i) * 4
    mad.lo.u32 %r1, %tx, %fi, %i;
    add.u32 %r1, %r1, %rW;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t1, %pMW, %t2;
    ld.global.f32 %m, [%t1];
    mul.f32 %m, %m, 0.9;
    fma.rn.f32 %m, %g, 0.1, %m;
    st.global.f32 [%t1], %m;
    add.u64 %t1, %pVW, %t2;
    ld.global.f32 %vv, [%t1];
    mul.f32 %vv, %vv, 0.999;
    mul.f32 %f1, %g, %g;
    fma.rn.f32 %vv, %f1, 0.001, %vv;
    st.global.f32 [%t1], %vv;
    div.full.f32 %mh, %m, %bc1;
    div.full.f32 %vh, %vv, %bc2;
    sqrt.approx.f32 %f1, %vh;
    add.f32 %f1, %f1, 0.00000001;
    div.full.f32 %dW, %mh, %f1;
    mul.f32 %dW, %lr, %dW;
ROW_APPLY:
    add.f32 %f1, %wOld, %dW;
    st.global.f32 [%t3], %f1;
    // atom delta[i] += wOld(pre-update) * dj  (CPU accumulates pre-update weights)
    mul.f32 %f1, %wOld, %dj;
    cvt.u64.u32 %t2, %i;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %bDdst, %t2;
    atom.global.add.f32 %f2, [%t2], %f1;
    add.u32 %i, %i, 1;
    bra ROW_LP;
ROW_DONE:
    // ---- bias: gb = dj + bdec * b[tid] ----
    add.u32 %r1, %rB, %tx;
    cvt.u64.u32 %t1, %r1;
    shl.b64 %t1, %t1, 2;
    add.u64 %t3, %pBB, %t1;
    ld.global.f32 %wOld, [%t3];
    fma.rn.f32 %g, %bdec, %wOld, %dj;
    @%useAdam bra ROW_ADAM_B;
    mul.f32 %dW, %lr, %g;
    bra ROW_APPLY_B;
ROW_ADAM_B:
    add.u64 %t2, %pMB, %t1;
    ld.global.f32 %m, [%t2];
    mul.f32 %m, %m, 0.9;
    fma.rn.f32 %m, %g, 0.1, %m;
    st.global.f32 [%t2], %m;
    add.u64 %t2, %pVB, %t1;
    ld.global.f32 %vv, [%t2];
    mul.f32 %vv, %vv, 0.999;
    mul.f32 %f1, %g, %g;
    fma.rn.f32 %vv, %f1, 0.001, %vv;
    st.global.f32 [%t2], %vv;
    div.full.f32 %mh, %m, %bc1;
    div.full.f32 %vh, %vv, %bc2;
    sqrt.approx.f32 %f1, %vh;
    add.f32 %f1, %f1, 0.00000001;
    div.full.f32 %dW, %mh, %f1;
    mul.f32 %dW, %lr, %dW;
ROW_APPLY_B:
    add.f32 %f1, %wOld, %dW;
    st.global.f32 [%t3], %f1;
ROW_MAXN:
    // ---- MaxNorm cap (3.0) over row tid, always applied ----
    mov.f32 %n2, 0.0;
    mov.u32 %i, 0;
MN_LP:
    setp.ge.u32 %ploop, %i, %fi;
    @%ploop bra MN_DONE;
    mad.lo.u32 %r1, %tx, %fi, %i;
    add.u32 %r1, %r1, %rW;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %pW, %t2;
    ld.global.f32 %f1, [%t2];
    fma.rn.f32 %n2, %f1, %f1, %n2;
    add.u32 %i, %i, 1;
    bra MN_LP;
MN_DONE:
    sqrt.approx.f32 %nn, %n2;
    setp.gt.f32 %over, %nn, %maxn;
    @!%over bra MN_SKIP;
    div.full.f32 %sc, %maxn, %nn;
    mov.u32 %i, 0;
MN_LP2:
    setp.ge.u32 %ploop, %i, %fi;
    @%ploop bra MN_SKIP;
    mad.lo.u32 %r1, %tx, %fi, %i;
    add.u32 %r1, %r1, %rW;
    cvt.u64.u32 %t2, %r1;
    shl.b64 %t2, %t2, 2;
    add.u64 %t2, %pW, %t2;
    ld.global.f32 %f1, [%t2];
    mul.f32 %f1, %f1, %sc;
    st.global.f32 [%t2], %f1;
    add.u32 %i, %i, 1;
    bra MN_LP2;
MN_SKIP:
PH1_DONE:
    bar.sync 0;
    // ---- phase 2: deriv multiply over fanIn ----
    mov.u32 %t, %tx;
PH2_LP:
    setp.ge.u32 %ploop, %t, %fi;
    @%ploop bra PH2_DONE;
    add.u32 %r1, %t, 0;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t2, %bDdst, %t3;
    ld.global.f32 %d, [%t2];
    // z = N[rAI + t], a = A[rAI + t]
    add.u32 %r1, %rAI, %t;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t2, %pN, %t3;
    ld.global.f32 %zz, [%t2];
    add.u64 %t2, %pA, %t3;
    ld.global.f32 %aa, [%t2];
    call (%dd), deriv_apx, (%zz, %aa, %aid);
    mul.f32 %d, %d, %dd;
    add.u32 %r1, %t, 0;
    cvt.u64.u32 %t3, %r1;
    shl.b64 %t3, %t3, 2;
    add.u64 %t3, %bDdst, %t3;
    st.global.f32 [%t3], %d;
    add.u32 %t, %t, %bd;
    bra PH2_LP;
PH2_DONE:
    bar.sync 0;
    setp.eq.u32 %p, %l, 0;
    @%p bra BWD_DONE;
    sub.u32 %l, %l, 1;
    bra BLAYER;
BWD_DONE:
    ret;
}
";
}
