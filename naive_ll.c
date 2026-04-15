/*
 * naive_ll.c  —  traditional schoolbook Lucas-Lehmer, pure-C, no GPU/MPI
 *
 * Algorithm: test M_p = 2^p - 1 using s_{k+1} = s_k^2 - 2 (mod M_p)
 * Big-integer representation: little-endian 64-bit limbs
 * Squaring: full schoolbook O(n²).  Reduction: one-shot Mersenne fold.
 *
 * Build: clang -O2 -D_CRT_SECURE_NO_WARNINGS naive_ll.c -o naive_ll.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── 64-bit limb big-integer, little-endian ─────────────────────── */
/* nl = ceil(p/64) limbs.  A product needs 2*nl limbs. */

/* Schoolbook multiply: prod[0..2nl-1] = a[0..nl-1] * b[0..nl-1]
   Uses 128-bit accumulator per output limb to keep it simple & correct. */
static void schoolbook_mul(const uint64_t *a, const uint64_t *b,
                            uint64_t *prod, int nl){
    memset(prod, 0, 2*nl*sizeof(uint64_t));
    for(int i=0;i<nl;i++){
        unsigned __int128 carry=0;
        for(int j=0;j<nl;j++){
            unsigned __int128 t = (unsigned __int128)a[i]*b[j]
                                + prod[i+j] + carry;
            prod[i+j]=(uint64_t)t;
            carry=t>>64;
        }
        /* propagate final carry */
        for(int k=i+nl;carry;k++){
            unsigned __int128 t=(unsigned __int128)prod[k]+carry;
            prod[k]=(uint64_t)t; carry=t>>64;
        }
    }
}

/* Mersenne reduction: x mod (2^p - 1).
   x has up to 2*nl limbs.  Result fits in nl limbs (stored back in x[0..nl-1]).
   Strategy: x mod M_p = (x & M_p) + (x >> p), repeat until < M_p.
   Because x < M_p^2, one fold reduces it to < 2*M_p, one more (or a compare)
   finishes. */
static void mersenne_reduce(uint64_t *x, int nx, int p){
    int nl=(p+63)/64;
    int hi_limb  = p/64;    /* word index where bit p lives  */
    int hi_shift = p%64;    /* bit within that word          */

    /* fold: lo = x & (2^p-1), hi = x >> p; x = lo + hi */
    /* We do at most 2 folds since x < M_p^2 < 2^(2p) */
    for(int fold=0;fold<3;fold++){
        /* extract hi (x >> p) into a temp array of nl limbs */
        uint64_t hi[512]={0}; /* p up to 32768, nl = p/64 <= 512 */
        int hi_words = nx - hi_limb;
        if(hi_words<=0) break;   /* already < 2^p: done */

        if(hi_shift==0){
            int copy = hi_words < nl ? hi_words : nl;
            for(int i=0;i<copy;i++) hi[i]=x[hi_limb+i];
        } else {
            for(int i=0;i<hi_words && i<nl;i++){
                hi[i] = x[hi_limb+i]>>hi_shift;
                if(hi_limb+i+1<nx) hi[i]|=x[hi_limb+i+1]<<(64-hi_shift);
            }
        }

        /* check if hi is all zero */
        int any=0; for(int i=0;i<nl;i++) if(hi[i]){any=1;break;}
        if(!any) break;

        /* zero out high portion in x */
        if(hi_shift==0){
            for(int i=hi_limb;i<nx;i++) x[i]=0;
        } else {
            x[hi_limb] &= (1ULL<<hi_shift)-1ULL;
            for(int i=hi_limb+1;i<nx;i++) x[i]=0;
        }

        /* x = x + hi (both fit in nl limbs) */
        unsigned __int128 c=0;
        for(int i=0;i<nl;i++){
            unsigned __int128 t=(unsigned __int128)x[i]+hi[i]+c;
            x[i]=(uint64_t)t; c=t>>64;
        }
        /* final carry: c might be 1, meaning x overflowed past 2^p,
           which means x = x - M_p + c, i.e. add c back (since 2^p ≡ 1 mod M_p) */
        if(c){
            unsigned __int128 cc=c;
            for(int i=0;i<nl && cc;i++){
                unsigned __int128 t=(unsigned __int128)x[i]+cc;
                x[i]=(uint64_t)t; cc=t>>64;
            }
        }
    }
    /* canonical reduction: if x == M_p, set x = 0 */
    int all_ff=1;
    for(int i=0;i<nl;i++){
        uint64_t mask = (i==hi_limb && hi_shift!=0) ? (1ULL<<hi_shift)-1ULL : ~0ULL;
        if(i>hi_limb) { if(x[i]) {all_ff=0;break;} continue; }
        if(x[i]!=mask){all_ff=0;break;}
    }
    if(all_ff) memset(x,0,nl*8);
}

/* Lucas-Lehmer test for M_p = 2^p - 1 */
static int lucas_lehmer(int p){
    if(p==2) return 1;
    int nl=(p+63)/64;
    uint64_t *s    = calloc(2*nl, 8);   /* current s, padded for product */
    uint64_t *prod = calloc(2*nl, 8);

    s[0]=4;  /* initial seed */

    for(int iter=0;iter<p-2;iter++){
        /* s = s^2 mod M_p */
        schoolbook_mul(s, s, prod, nl);
        /* copy product back, reduce */
        memcpy(s, prod, 2*nl*8);
        mersenne_reduce(s, 2*nl, p);

        /* s = s - 2 mod M_p
         * s ∈ [0, M_p-1].  If s < 2, add M_p first so the subtraction
         * won't underflow past zero.  Adding M_p to {0,1} gives {M_p-1,M_p}
         * which still fits in nl limbs (nl*64 ≥ p+1). */
        {
            int hi_limb2=p/64, hi_shift2=p%64;
            /* check s < 2: limbs 1..nl-1 all zero, limb 0 < 2 */
            int s_lt_2 = (s[0] < 2);
            for(int i=1; i<nl && s_lt_2; i++) if(s[i]) s_lt_2=0;

            if(s_lt_2){
                /* s += M_p  (no carry out: s+M_p ≤ 1+(2^p-1)=2^p < 2^(nl*64)) */
                unsigned __int128 c=0;
                for(int i=0;i<nl;i++){
                    uint64_t mp = (i<hi_limb2)?~0ULL:
                                  (i==hi_limb2&&hi_shift2)?(1ULL<<hi_shift2)-1ULL:0ULL;
                    unsigned __int128 t=(unsigned __int128)s[i]+mp+c;
                    s[i]=(uint64_t)t; c=t>>64;
                }
            }
            /* now s ≥ 2, safe to subtract without underflow */
            uint64_t borrow2=2;
            for(int i=0;i<nl;i++){
                uint64_t prev=s[i]; s[i]-=borrow2;
                borrow2=(prev<borrow2)?1:0;
                if(!borrow2) break;
            }
        }
    }

    /* result: s == 0 means M_p is prime */
    int zero=1;
    for(int i=0;i<nl;i++) if(s[i]){zero=0;break;}

    free(s); free(prod);
    return zero;
}

/* ── millisecond timer ── */
static double ms_now(void){
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}

int main(int argc, char **argv){
    int primes[]={2,3,5,7,13,17,19,31,61,89,107,127,
                  521,607,1279,2203,2281,3217,4253,4423,
                  9689,9941,11213,19937};
    int np=(int)(sizeof(primes)/sizeof(primes[0]));

    /* Single-argument mode: benchmark one p */
    if(argc==2){
        int p=atoi(argv[1]);
        double t0=ms_now();
        int r=lucas_lehmer(p);
        double dt=ms_now()-t0;
        printf("naive_ll M_%d = %s  (%.1f ms)\n",p,r?"PRIME":"COMPOSITE",dt);
        return !r;
    }

    printf("╔══ TRADITIONAL SCHOOLBOOK LL (naive_ll.c) ══════════════════╗\n");
    printf("  64-bit limbs, schoolbook O(n^2) squaring, no GPU\n");
    printf("  %-8s  %-9s  %10s  %8s\n","p","result","time_ms","digits");
    printf("  %-8s  %-9s  %10s  %8s\n","--------","---------","----------","--------");

    for(int i=0;i<np;i++){
        int p=primes[i];
        double t0=ms_now();
        int r=lucas_lehmer(p);
        double dt=ms_now()-t0;
        int digits=(int)(p*0.30103)+1;
        printf("  %-8d  %-9s  %10.1f  %8d\n",p,r?"PRIME":"COMPOSITE",dt,digits);
        fflush(stdout);
        if(dt>120000.0){
            printf("  (stopping — beyond this takes >2min per test)\n");
            break;
        }
    }
    printf("╚════════════════════════════════════════════════════════════╝\n");
    return 0;
}
