//=== EXTRA SOUNDS FOR FREE USE ALONGSIDE THE MATRIX SOUNDS ===\\

// ---------------------------------------------------------------- *
//DEFINITION OF SOUNDS

(
(
Ndef(\lowpad, {
	var env, snd, local, tone, imp, snd2, rand;

	snd = SinOsc.ar(\freq.ar(150)) * 0.8;//2

	// ----- Feeback -----
	local = LocalIn.ar(1) * 0.7;
	local = DelayN.ar(local, 0.6, 0.5);
	local = PitchShift.ar(local, 0.8, SinOsc.ar(2, 0.75, 0.6, 0.5) * 0.6, \dispers.ar(4));
	local = (local + snd).tanh;
	LocalOut.ar(local);
	// ---------------

	imp = Impulse.ar(0.5, 0.4);
	local = SinOsc.ar(\mainfreq.ar(50) + (local * 20)) * 0.3 * Hasher.ar(TRand.ar(0, 200, imp));//
	local = GVerb.ar(local, 10, 2, 0.2);
	local = LPF.ar(local, \lpf.ar(500));//3
	local = HPF.ar(local, \hpffreq.ar(60));//1

	local = local * 2 * \amp.ar(1);
	local = local.sum.sanitize.tanh;
})
);

(
Ndef(\sprinkle, {
	var snd, env, local, in, snd2, rand, envelope, imp;

	imp = Impulse.ar(\trig.ar(0.2), 0.4);//5
	envelope = EnvGen.ar(Env.perc(\envatt.ar(0.01), \envrel.ar(0.075)), imp);//1
	rand = TIRand.ar(0, 150, LFNoise0.ar(1));
	snd2 = SinOsc.ar(\freq.ar(100) + rand) * 2 * envelope;//4

	// ----- Feedback -----
	local = LocalIn.ar(2) * 0.7;
	local = HPF.ar(local, 200);
	local = PitchShift.ar(local, 0.3, \pratio.ar(5.5), 4);//2
	local = (local + snd2).tanh;
	LocalOut.ar(local);
	// ---------------

	local = LPF.ar(local, SinOsc.ar(2).range(200, \lpf.ar(500)));//3
	local = FreeVerb.ar(local, 0.46, 0.8, 0.3);

	local = HPF.ar(local, 800);
	local = local * 3 * \amp.ar(1);
	local = local.sum.sanitize.tanh;
})
);

(
Ndef(\bassy, {
	var snd, freq, verb;

	freq = SelectX.ar(\filtermod.ar(0), [DC.ar(100), SinOsc.ar(0.4 + ((SinOsc.ar(0.4) * 0.5)) * SinOsc.ar(0.6)).range(50, 100)]);//5

	snd = SinGrain.ar(Dust.ar(\trigfreq.ar(1)), \duration.ar(1), \basefreq.ar(35));//d3, t6
	snd = snd * SinOscFB.ar(freq, mul: 0.5);
	snd = FreqShift.ar(snd, \shift.ar(0));//2

	snd = GVerb.ar(snd, 20, 1.5, drylevel: \drylevel.ar(0));//6
	snd = Decimator.ar(snd, 44100, \bits.ar(24));//bits -> 3 //4
	snd = LPF.ar(snd, \lpffreq.ar(300));
	snd = HPF.ar(snd, \hpffreq.ar(500));//1

	snd = snd * 0.6 * \amp.ar(1);
	snd.sum.sanitize.tanh;
})
);

(
Ndef(\pad, {
	var snd, mod1, mod2, verb;

	mod1 = Select.ar(\mod1.ar(0), [DC.ar(5000), LFNoise0.ar(0.3).linlin(-1, 1, 100, 3000)]);

	snd = {Dust.ar(\density.ar(2))}!15; //1
	snd = Ringz.ar(snd/30, [60, 65, 70, 75, 80].midicps ++ ({1000.0.rand + mod1}!10), 0.3, (1!5)++(0.3!10));
	snd = SelectX.ar(\pulsemod.ar(0), [snd, snd * Pulse.ar(\pulsefreq.ar(90))]);//4
	snd = LPF.ar(snd, \lpffreq.ar(200));//2

	verb = GVerb.ar(snd.sum, 20, 1, drylevel: 0.5) * SinOsc.ar(\verbmod.ar(0.5));
	snd = (snd * \dryverb.ar(1)) + (verb * \wetverb.ar(0));//3
	snd = SelectX.ar(\bpf.ar(0), [snd, BPF.ar(snd, {2000.rand + 500}!5, 0.3) * 2.2]);//5

	snd = snd * 0.6 * \amp.ar(3);
	snd = snd.sum.sanitize.tanh;
})
);

(
Ndef(\texture, {
	var mod, snd;

	mod = SinGrain.ar(Impulse.ar(Latch.ar(SinOsc.ar(1.3), Impulse.ar(0.5)).range(0.3, 1)), 3, 25);
	mod = mod * Pulse.ar(\pulsemod.ar(4), mul: 0.5).lag(0.1);//3
	snd = FBSineC.ar(mod.range(200, 500), \im.ar(0.7));//-
	snd = RLPF.ar(snd, {\filterfreqlo.ar(3000).rand + 1000}!5, 0.2) / 5;//5

	snd = GVerb.ar(snd.sum, 10, 2);
	snd = snd * Env.perc(\envatt.ar(0.5), \envrel.ar(1.0)).ar(0, Impulse.ar(0.3));//2
	snd = CombN.ar(snd, 0.5, 0.1, \decaytime.ar(0.3));//4
	snd = HPF.ar(snd, \hpf.ar(550));//1

	snd = snd * 0.3 * \amp.ar(1);
	snd.sum.sanitize.tanh;
})
);
)