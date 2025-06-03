//=== MODULATION AND SOURCE SIGNALS FOR PLAYFIELD/MODULATION MATIRX 2 ===\\

// ---------------------------------------------------------------- *
// COLUMNS - modulation signals
(
(
Ndef(\sine_2, {
	var mod;
	mod = SinOsc.ar(0.2) * SinOsc.ar(LFDNoise0.ar(0.3).linlin(-1, 1, 1.0, 20.0));
})
);

(
Ndef(\sine2_2, {
	var mod;
	mod = SinOsc.ar(1) * SinOsc.ar(0.2);
})
);

(
Ndef(\sine44, {
	var mod;
	mod = SinOsc.ar(2);
})
);

(
Ndef(\sineFB_2, {
	var mod;
	mod = SinOscFB.ar(75, SinOsc.ar(0.5).range(65, 85));
})
);

(
Ndef(\sineGrain_2, {
	var mod;
	mod = SinGrain.ar(Impulse.ar(10.0), 3, 20);
	// mod = mod.sum / 3;
})
);

(
Ndef(\sineLatch_2, {
	var mod;
	mod = Latch.ar(SinOsc.ar(0.4), Impulse.ar(0.5));
})
);

(
Ndef(\sineTone_2, {
	var freq, mod;
	//counters \sine2
	freq = Demand.ar(Impulse.ar(0.3), 0, Dseq([900, 650, 700], inf));
	mod = SinOsc.ar(freq);
})
);

(
Ndef(\sineTone1_2, {
	var freq, mod;
	freq = Demand.ar(Impulse.ar(0.3), 0, Dseq([550, 750, 600], inf));
	mod = SinOsc.ar(freq);
})
);

(
Ndef(\tri_2, {
	var mod;
	mod = LFTri.ar(0.2);
})
);

(
Ndef(\lftri_2, {
	var mod;
	mod = LFTri.ar(0.01).range(0.0001, 0.01);
})
);

(
Ndef(\lfpulse_2, {
	var mod;
	mod = LFPulse.ar(\freq.ar(60));
})
);

(
Ndef(\pulse44, {
	var mod;
	mod = Pulse.ar(3 + (SinOsc.ar(0.3) * 2));
})
);

(
Ndef(\amPulse_2, {
	var mod;
	mod = Pulse.ar(200) * SinOsc.ar(\modfreq.ar(0.2)).range(0.0, 0.9);
})
);

(
Ndef(\noiseT_2, {
	var mod;
	mod = LFNoise0.ar(0.3);
})
);

(
Ndef(\noiseT2_2, {
	var mod;
	mod = LFNoise0.ar(2.0);
})
);


(
Ndef(\deepNoise_2, {
	var mod;
	mod = LFDNoise0.ar(2).range(LFDNoise0.ar(5).range(0, 10), LFDNoise0.ar(5).range(2, 20));
})
);

(
Ndef(\env_2, {
	var mod;
	mod = Env.perc(0.01, 1.0).ar(0, Impulse.ar(0.5)) * 10;
})
);

(
Ndef(\ay_2, {
	var mod;
	mod = AY.ar(2000, 3000, 600);
})
);

(
Ndef(\demand_22, {
	var mod;
	mod = Demand.ar(LFNoise0.ar(2, 0.5, 0.5).round, 0, Dbrown(190, 300, 150, inf));
})
);

(
Ndef(\demand2_2, {
	var mod;
	mod = Demand.ar(LFNoise0.ar(2, 0.5, 0.5).round, 0, Dbrown(300, 1000, 100, inf));
})
);

(
Ndef(\demand3, {
	var mod;
	mod = Demand.ar(LFNoise0.ar(2, add:1).round, 0, Dseq([400, 900, 550], inf))
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
Ndef(\src1_2, {
	var mod1, mod2, mod3, mod4, mod5, snd, fshift, pan, ele, azi;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(50), Ndef.ar(\noiseT_2).linlin(-1, 1, 50, 2500)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(0), Ndef.ar(\sineFB_2).abs]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(500), Ndef.ar(\sine_2).range(50, 900)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(1), Ndef.ar(\noiseT2_2).abs.round]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\amPulse_2)]);

	snd = Blip.ar(mod1 + (mod2 *  \noisefreq.ar(50)), mod3) * mod4; //FREQ (mod1), FM (mod2), NUMHARM (mod3), AM (mod4)
	snd = PitchShift.ar(snd, 0.5, 0.5, \disp.ar(5.0));

	snd = (snd * \dry.ar(1)) + (GVerb.ar(snd.sum, 10, \decay.ar(0.2)) * \verbOnOff.ar(0));
	// snd = LPF.ar(snd, 5000);
	snd = snd * mod5; //AMP (mod5)

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.7;
	// snd = snd.sum.sanitize.tanh!2
})
);

(
Ndef(\src2_2, {
	var mod1, mod2, mod3, mod4, mod5, snd, azi, ele, pan;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(1777), Ndef.ar(\noiseT_2).linexp(-1, 1, 500, 2000)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(5), Ndef.ar(\sine_2).range(1, 6)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(0), Ndef.ar(\tri_2).range(0, 0.6)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(0), Ndef.ar(\env_2)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(3), Ndef.ar(\noiseT_2).range(0.5, 4)]);

	snd = AY.ar(mod1);//TONEA (mod1)
	snd = snd * ((Pulse.ar(mod2, 0.5, 0.5) * LFSaw.ar(\modamp.ar(0.5))) + mod3);//FREQ (mod2), AMAOFFSET (mod3)
	snd = HPF.ar(snd, \hpffreq.ar(600) + (mod4 * 2000));//FM (mod4)
	snd = PitchShift.ar(snd, 0.2, mod5, \pitchdisp.ar(0.005));//PITCHRATIO (mod5)
	snd = snd * (0.9+(Pulse.ar(\pulsefreq.ar(3.0), 0.5, \extraPulse.ar(0.0))));

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 2;
	// snd = snd.sum.sanitize.tanh!2
})
);

(
Ndef(\src3_2, {|tSpeed = 7.0|
	var mod1, mod2, mod3, mod4, mod5, dur, snd, local, pan, azi, ele;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(0.5), Ndef.ar(\noiseT_2).linexp(-1, 1, 0.3, 10)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(50), Ndef.ar(\sineTone_2).range(10, 500)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(123), Ndef.ar(\demand_22)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(2000), Ndef.ar(\sine_2).range(500, 1500)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\lfpulse_2) * 0.6]);

	dur = Demand.ar(Impulse.ar(0.5), 0, Drand([0.4, 3, 0.8, 0.09, 2], inf));

	snd = FMGrain.ar(Impulse.ar(mod1), dur, mod2, mod3, \modfilter.ar(5)).clip;//FREQ (mod1), CARFREQ (mod2), MODFREQ (mod3)
	snd = LPF.ar(snd, mod4);//FREQ (mod4)
	snd = (snd * \dry.ar(1)) + (CombN.ar(snd, 2.0, 0.1, \decay.ar(3)) * \comblvl.ar(0));
	snd = snd * mod5;//AM (mod5)

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.2;
	// snd = snd.sum.sanitize.tanh!2

})
);

(
Ndef(\src4_2, {
	var mod1, mod2, mod3, mod4, mod5, env, snd;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(0.2), Ndef.ar(\noiseT_2).range(0.4, 9)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(0), Ndef.ar(\demand3)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(1), Ndef.ar(\sine44)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(0), Ndef.ar(\sineLatch_2).range(0.0, 0.9)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\pulse44)]);

	env = Env.perc(0.1, 0.4).ar(0, Pulse.ar(mod1).lag(0.2));//FREQ (mod1)
	snd = SinGrain.ar(Impulse.ar(\newtrigger.ar(1)), \duration.ar(2), 600 + (LFTri.ar(500) * mod2)) * mod3;//AM (mod2),AM (mod3)
	snd = snd * env;
	snd = PitchShift.ar(snd * (mod5), 0.5, 0.4, mod4);//PITCHDISP (mod4), AM (mod5)
	snd = GVerb.ar(snd.sum, 50, 3);
	snd = HPF.ar(snd, \hpffreq.ar(1000));

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10),\ele.ar(0).lag(15)) * 0.4;
})
);

(
Ndef(\src5_2, {
	var mod1, mod2, mod3, mod4, mod5, env, penv, freqs, snd, verbMod, reverb, pan;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(350), Ndef.ar(\demand3)]);
	mod2 = Select.ar(\mod2.ar(0), [DC.ar(0), Ndef.ar(\sine2_2)]);
	mod3 = Select.ar(\mod3.ar(0), [DC.ar(600), Ndef.ar(\demand2_2)]);
	mod4 = Select.ar(\mod4.ar(0), [DC.ar(400), Ndef.ar(\noiseT_2).range(250, 2500).lag(10)]);
	mod5 = Select.ar(\mod5.ar(0), [DC.ar(1), Ndef.ar(\amPulse_2)]);

	snd = Saw.ar(mod1 * [1, 1.01] + (mod2 * \modval.ar(150))) * Pulse.ar(\trig.ar(2.0));//FREQ (mod1), FM (mod2)
	snd = FreqShift.ar(snd, mod3);//FREQ (mod3)
	snd = Resonz.ar(snd, mod4, 2) * mod5;//FREQ (mod4), AMP (mod5)

	verbMod = SinOsc.ar(\verbmodfreq.ar(0.5), mul: \verbmodlvl.ar(1));
	reverb = GVerb.ar(snd.sum, \room.ar(5), 2) * (1+verbMod);
	snd = (snd * \dry.ar(1)) + (reverb * \wetlvl.ar(0));
	snd = (snd * 0.2).fold(-0.99, 0.99);

	snd = VBAP.ar(a.numSpeakers, snd.sum.sanitize.tanh, b.bufnum, \azi.ar(0).lag(10), \ele.ar(0).lag(15)) * 0.5;
})
);

// MIX SIGNALS AND OUTPUT THEM
(
Ndef(\mix_2, {
	var src1, src2, src3, src4, src5, fb, out;

	src1 = Ndef.ar(\src1_2) * (\src1.ar(0.0) * \liveamp1.ar(1));
	src2 = Ndef.ar(\src2_2) * (\src2.ar(0.0) * \liveamp2.ar(1));
	src3 = Ndef.ar(\src3_2) * (\src3.ar(0.0) * \liveamp3.ar(1));
	src4 = Ndef.ar(\src4_2) * (\src4.ar(0.0) * \liveamp4.ar(1));
	src5 = Ndef.ar(\src5_2) * (\src5.ar(0.0) * \liveamp5.ar(1));
	//=============

	out = src1 + src2 + src3 + src4 + src5 + DC.ar(0);
	out = LPF.ar(out, 17000);
	out = out * \amp.ar(1).lag(0.2);

	Limiter.ar(out.sanitize, 0.999);
})
);

)
Ndef(\mix_2).play;
Ndef(\mix_2).stop(0.1);

Ndef.clear;
Ndef.dictFor(s).fadeTime = 0.2;

// ---------------------------------------------------------------- *
// SET DEFAULT POSITIONING OF SOUNDS
(
Ndef(\src1_2).set(\azi, 90);
Ndef(\src2_2).set(\azi, 40);
Ndef(\src3_2).set(\azi, -60);
Ndef(\src4_2).set(\azi, 0);
Ndef(\src5_2).set(\azi, 45);

Ndef(\src1_2).set(\ele, 20);
Ndef(\src2_2).set(\ele, 20);
Ndef(\src3_2).set(\ele, 20);
Ndef(\src4_2).set(\ele, 20);
Ndef(\src5_2).set(\ele, 20);
)
