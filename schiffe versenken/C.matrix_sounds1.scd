//=== MODULATION AND SOURCE SIGNALS FOR PLAYFIELD/MODULATION MATIRX 1 ===\\

// ---------------------------------------------------------------- *
// COLUMNS - modulation signals
(
(
Ndef(\sine, {
	var mod;

	mod = SinOsc.ar(0.3) * SinOsc.ar(0.5);
})
);

(
Ndef(\sine2, {
	var freq, mod;
	// freq = Demand.ar(Impulse.ar(0.3), 0, Dseq([150, 300, 90], inf));
	mod = SinOsc.ar(4);//5
})
);

(
Ndef(\sineTone, {
	var freq, mod;

	freq = Demand.ar(Impulse.ar(0.3), 0, Dseq([150, 300, 90], inf));
	mod = SinOsc.ar(freq);
})
);

(
Ndef(\sineTone1, {
	var freq, mod;
	//counters \sine2
	freq = Demand.ar(Impulse.ar(0.3), 0, Dseq([230, 130, 160], inf));
	mod = SinOsc.ar(freq);
})
);

(
Ndef(\fmSine, {
	var mod;

	mod = SinOsc.ar(0.2 + (LFNoise0.ar(0.5).range(0, 1) * 1));
})
);

(
Ndef(\sineGrain, {
	var mod;
	mod = SinGrain.ar(Impulse.ar(10.0), 3, 2500); //2500
	// mod = mod.sum / 3;
})
);

(
Ndef(\tri, {
	var mod;

	// demand = Demand.ar(Impulse.ar(1.0), 0, Pseq([60, 65, 70].midicps, inf));
	mod = LFTri.ar(10);
})
);

(
Ndef(\lfpulse, {
	var mod;
	LFPulse.ar(2.5) * SinOsc.ar(0.5);
})
);

(
Ndef(\pulse, {
	var mod;

	mod = Pulse.ar(3000);
})
);

(
Ndef(\noise, {
	var mod;
	LFDNoise0.ar(0.3);
})
);

(
Ndef(\noiseT, {
	var mod;

	mod = LFNoise0.ar(0.5);
	//auch für am gut
})
);

(
Ndef(\deepNoise, {
	var mod;
	//range 0 - 5
	mod = LFDNoise0.ar(200).range(LFDNoise0.ar(10).range(3, 7), LFDNoise0.ar(10).range(4, 9)).round;
})
);
)

// ---------------------------------------------------------------- *
// ROWS - source/carrier signals

// 5x5 field = 5 source signals with 5 modulation possibilities each
// modulation destinations and signals are different within each Ndef
// when modulation is turned off/or not turned on yet, there is a default value
(
(
Ndef(\src1_1, {
	var mod1, mod2, mod3, mod4, mod5, freq, snd, fizz, out, azi, ele;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(1), Ndef.ar(\fmSine).linexp(-1, 1, 0.2, 2)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(150), Ndef.ar(\sine2).linexp(-1, 1, 100, 500)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(0.1), Ndef.ar(\tri).linlin(-1, 1, 0.1, 2)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(0), Ndef.ar(\tri) * 3]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(0), Ndef.ar(\noiseT).linlin(-1, 1, 0, 1).round]);

	freq = Demand.ar(Impulse.ar(5), 0, Dbrown(\demlow.ar(1.0), \demhigh.ar(10.0), \demstep.ar(0.5), inf));
	snd = Impulse.ar(freq / mod1);//IMP (mod1)

	snd = Ringz.ar(snd, mod2, mod3); //FREQ (mod2), DECAY (mod3)
	snd = BPF.ar(snd, SinOsc.ar(2.0).range(\freqlow.ar(500), \freqhigh.ar(1000)));

	fizz = LFNoise0.ar(5); //FREQ (mod4)
	fizz = fizz + (SinOsc.ar(500, mul:0.5) * Env.perc(0.001, 0.1).ar(0, fizz) * mod4);
	snd = snd * 0.2;
	fizz = fizz * 0.2;


	out = SelectX.ar(mod5.lag(1), [snd, fizz]); //WHICH (mod5)

	//.lag azimut&elevation in order to hear the movement on position change
	out = VBAP.ar(a.numSpeakers, out.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.15;
	// out.sum.sanitize.tanh!2;
})
);

(
Ndef(\src2_1, {
	var mod1, mod2, mod3, mod4, mod5, t1, t2, trig, penv, env, snd, out, azi, ele;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(0.5), Ndef.ar(\noiseT).linexp(-1, 1, 5, 10).round]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(2.0), Ndef.ar(\sine).linexp(-1, 1, 5, 250)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(10000), Ndef.ar(\fmSine).linexp(-1, 1, 1000, 5000)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(400), Ndef.ar(\sineTone).linlin(-1, 1, 100, 1000).round]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\pulse)]);

	t1 = Impulse.ar(mod1); //FREQ (mod1)
	t2 = LFNoise1.ar(mod2, 0.5, 0.5).round; //FREQ (mod2)
	trig = Select.ar(LFNoise0.ar(1.0, 0.5, 0.5).round, [t1, t2]);

	penv = Env.perc(\penvatt.ar(0.01), 0.01, curve: -9).ar(0, trig);
	env = Env.perc(0.01, \envrel.ar(0.05)).ar(0, trig);
	snd = SinOsc.ar(\freq.ar(150) + (penv * mod3)) * env; //PENV MODFREQ (mod3)
	snd = BPF.ar(snd, mod4) * mod5;//FREQ (mod4), AMP (mod5)

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.04;
	// snd.sum.sanitize.tanh!2;
})
);

(
Ndef(\src3_1, {|tSpeed = 7.0|
	var mod1, mod2, mod3, mod4, mod5, env, snd, out, azi, ele, mod;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(0.001), Ndef.ar(\fmSine).linlin(-1, 1, 0.001, 0.09)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(tSpeed), Ndef.ar(\deepNoise)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(-20), Ndef.ar(\noiseT).linexp(-1, 1, -20, -5).round]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(500), Ndef.ar(\sine).linlin(-1, 1, 50, 1500)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\sineTone1).range(0, 0.5)]);

	env = Env.perc(0.001, mod1).ar(0, Impulse.ar(mod2));//ENVRELEASSE (mod1), TSPEED (mod2)
	snd = ClipNoise.ar() * env;

	snd = Ringz.ar(snd, TChoose.ar(Impulse.ar(tSpeed), DC.ar([100, 175, 200])), \decayt.ar(0.6));
	snd = (snd * mod3.dbamp).fold(\foldlow.ar(-0.9), 0.9); //DBAMP (mod3)
	snd = RLPF.ar(snd, mod4, \filterres.ar(0.8));//FREQ (mod4)
	// snd = BPF.ar(snd, mod4, 0.2); //freq = XLine.ar(0.5, 300, 20)//SINOS
	snd = snd * mod5;//AM (mod5)

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.02;
	// snd.sum.sanitize.tanh!2;
})
);

(
Ndef(\src4_1, {
	var mod1, mod2, mod3, mod4, mod5, am, freqs, snd, out, azi, ele;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(2000), Ndef.ar(\fmSine).range(1900, 2000)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(4), Ndef.ar(\noiseT).linlin(-1, 1, 3, 8)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(0.3), Ndef.ar(\sine).range(0.1, 0.5)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(0), Ndef.ar(\lfpulse).range(0.0, 0.1)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\sineGrain)]);

	am = 1;
	freqs = Demand.ar(LFNoise0.ar(4, 0.5, 0.5).round, 0, Dseq([60, 64, 67, 69, 70].midicps, inf));

	snd = CuspN.ar(100 * [1, 1.01], 1);
	snd = Formlet.ar(snd, mod1, 0.3, 0.9).fold(-0.9, \foldhigh.ar(0.9));//FREQ (mod1)
	snd = snd * (Pulse.ar(\trig.ar(0.2)) * am);
	snd = Ringz.ar(snd, \resfreq.ar(100), 0.9) * Env.perc(\envatt.ar(0.001), 5.0).ar(0, Pulse.ar(mod2));//PULSEFREQ (mod2)
	snd = (snd * \drylevel.ar(1.0)) + (CombN.ar(snd, 1.0, mod3, 1.0) * \dellevel.ar(0.1) * mod4);//DELAYTIME (mod3), DELAYLEVEL AM (mod4)
	snd = LPF.ar(snd, \filterfreq.ar(1000));

	snd = (snd * 0.5) * mod5;//AMP (mod5)
	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.01;
	// snd.sum.sanitize.tanh!2 * 0.2;
})
);

(
Ndef(\src5_1, {
	var mod1, mod2, mod3, mod4, mod5, trig, amp, snd, out, azi, ele;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(100), Ndef.ar(\noiseT).linlin(-1, 1, 50, 150)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(2.5), Ndef.ar(\noise).linlin(-1, 1, 2.5, 6).round]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(0), Ndef.ar(\sine)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(1), Ndef.ar(\sineTone).range(0, 0.08)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\sine)]);

	trig = LFNoise0.ar(5);
	amp = Demand.ar(trig, 0, Drand([1, 1, 0], inf)).lag(0.5);

	snd = LatoocarfianN.ar(mod1, 0.5, mod2, \diff.ar(0.5), 0.3);//FREQ (mod1), B (mod2)
	snd = BPF.ar(snd, \basefreq.ar(50) + (mod3 * 50), 0.4);//FM FREQ (mod3)
	snd = (snd * (amp * mod4)).fold(-0.9, \foldhigh.ar(0.8)); //DBAMP (mod4)
	snd = RHPF.ar(snd, 200, 0.1);
	snd = LPF.ar(snd, \lpffreq.ar(3000));

	snd = snd * mod5 * amp; //AM (mod5)
	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.1;
	// snd.sum.sanitize.tanh!2;
})
);

// MIX SOURCE SIGNALS AND OUTPUT THEM
(
Ndef(\mix_1, {
	var src1, src2, src3, src4, src5, src6, fb, out;

	src1 = Ndef.ar(\src1_1) * (\src1.ar(0.0) * \liveamp1.ar(1));
	src2 = Ndef.ar(\src2_1) * (\src2.ar(0.0) * \liveamp2.ar(1));
	src3 = Ndef.ar(\src3_1) * (\src3.ar(0.0) * \liveamp3.ar(1));
	src4 = Ndef.ar(\src4_1) * (\src4.ar(0.0) * \liveamp4.ar(1));
	src5 = Ndef.ar(\src5_1) * (\src5.ar(0.0) * \liveamp5.ar(1));
	//=============

	out = src1 + src2 + src3 + src4 + src5 + DC.ar(0);
	out = LPF.ar(out, 17000);
	out = out * 20 * \amp.ar(1).lag(0.2);

	Limiter.ar(out.sanitize.tanh, 0.999);
})
);
)
Ndef(\mix_1).play;
Ndef(\mix_1).stop(0.1);

Ndef.clear;
Ndef.dictFor(s).fadeTime = 0.2;

// ---------------------------------------------------------------- *
// SET DEFAULT POSITIONING OF SOUNDS
(
Ndef(\src1_1).set(\azi, 45);
Ndef(\src2_1).set(\azi, -90);
Ndef(\src3_1).set(\azi, -60);
Ndef(\src4_1).set(\azi, 90);
Ndef(\src5_1).set(\azi, -180);

Ndef(\src1_1).set(\ele, 20);
Ndef(\src2_1).set(\ele, 20);
Ndef(\src3_1).set(\ele, 20);
Ndef(\src4_1).set(\ele, 20);
Ndef(\src5_1).set(\ele, 20);
)
