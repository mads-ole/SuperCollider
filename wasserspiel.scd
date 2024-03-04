s.boot;
s.plotTree;
s.meter;
//ggf. notwendig
s.options.memSize = 8192 * 32;


//Buffer initialisieren
//gerne auch Austauschen der Buffer
(
(
~b = Buffer.read(s, "/Users/madsoleclasen/Desktop/Flötenkreis--BtW1xjNkFTo.wav");

~c = Buffer.read(s, "/Users/madsoleclasen/Documents/SAMPLES/good drums/The Kount/The Kount - Sample Pack Vol. 2/THE_KOUNT_tonal/THE_KOUNT_tonal_loops/THE_KOUNT_82_rhodes_loop_mystery_Fmin.wav");

~d = Buffer.read(s, "/Users/madsoleclasen/Documents/SAMPLES/good drums/The Kount/The Kount - Sample Pack Vol. 2/THE_KOUNT_tonal/THE_KOUNT_tonal_loops/THE_KOUNT_118_synth_loop_sneaky_Dmin.wav");

~e = Buffer.read(s, "/Users/madsoleclasen/Documents/SAMPLES/good drums/The Kount/The Kount - Sample Pack Vol. 2/THE_KOUNT_tonal/THE_KOUNT_tonal_loops/THE_KOUNT_104_flute_loop_willpower.wav");

~f = Buffer.read(s, "/Users/madsoleclasen/Documents/SAMPLES/good drums/The Kount/The Kount - Sample Pack Vol. 2/THE_KOUNT_tonal/THE_KOUNT_tonal_loops/THE_KOUNT_141_synth_loop_lost_G.wav");
);

(
~array = [
	~b,
	~c,
	~d,
	~e,
	~f
]
);
)


//SynthDefs initialisieren
(
//Ausgangssignal
(
SynthDef(\snd, {
	arg out_fx = 8, out_chain = 50;
	var trig, lineTo0, env1, env2, env, noise, buf, sig;

	trig = Impulse.ar(\impulse_freq.ar(1.0).lag(50) * [0.25, 0.3]);
	lineTo0 = Line.ar(1.0, 0.0, \noise_time.ir(20.0));

	env1 = EnvGen.ar(Env([lineTo0, 1, lineTo0], [0.0001, 0.5, curve: \exp]), trig, doneAction: 0);
	env2 = EnvGen.ar(Env.perc(\attack.ar(0.001), 1.0), trig, doneAction: 0);

	env = Select.ar(\envelope.ar(0), [env1, env2]);

	noise = WhiteNoise.ar(env);

	buf = PlayBuf.ar(2, Select.kr(TRand.ar(0, 4, Impulse.ar(5)), ~array)) * env;

	sig = SelectX.ar(Line.ar(0.0, 1.0, 50.0), [noise, buf]);
	sig = (\buf_fadeout.ar(1.0).lag(120) * sig) + (\noise_fadein.ar(0.0).lag(120) * noise);

	Out.ar(out_fx, sig * (-45.dbamp));
	Out.ar(out_chain, sig * (-45.dbamp));
}).add;
);

//FX-Synth 1
(
SynthDef(\fb4, {
	arg out_fx = 8, out_chain = 50; //->18
	var sig, fbIn, local, snd, reson;
	sig = In.ar(16, 2);
	fbIn = In.ar(30);

	local = LocalIn.ar(4) * 0.99 + sig;
	snd = DelayN.ar(local, 0.5, fbIn.range(0.0, 2.0) * [0.45, 0.3, 0.35, 0.25]);
	snd = BPF.ar(snd, [500, 700, fbIn.range(1000, 1600), 1900], 0.1);

	reson = Ringz.ar(snd, (LorenzL.ar(1)+70*50).range(50, 90), 3.0, 0.5);
	snd = SelectX.ar(\resonance.ar(0), [snd, reson]);

	LocalOut.ar(snd);

	Out.ar(30, snd * 50.dbamp);
	Out.ar(out_fx, snd * 50.dbamp);
	Out.ar(out_chain, snd * 50.dbamp);
}).add;
);


//FX-Synth 2.1
(
SynthDef(\fb2, {
	arg out_fx = 8, out_chain = 50; //->22
	var sig, fbIn, snd1, snd2, snd;
	sig = In.ar(18, 4);
	fbIn = In.ar(35);

	sig = sig !2;

	snd1 = PitchShift.ar(sig[0], 0.5, Saw.ar(0.33, 0.5, 0.5), Line.ar(0.0, 2.5, 20.0) + fbIn.range(0.1, 2.5), 0.25);

	snd2 = sig[1] * Saw.ar(SinOsc.ar(0.25).range(1.0, 11.0 + fbIn.exprange(0.0, 10.0))).range(0.0, 1.0);


	snd = snd1 + (snd2 * \tickAmp.ar(0.5).lag(10));
	// snd = [snd1, snd2];
	snd = DelayN.ar(snd, 2.0, 1.5);
	snd = Splay.ar(snd);

	Out.ar(\fb_out2.ar(35),  snd * 0.dbamp);
	Out.ar(out_fx, snd * 10.dbamp);
	Out.ar(out_chain, snd * 10.dbamp);
}).add;
);


//FX-Synth 2.2
(
SynthDef(\fb22, {
	arg out_fx = 8, out_chain = 50; //->24
	var sig, fbIn, snd1, env2, snd2, snd;
	sig = In.ar(22, 4); //22
	fbIn = In.ar(40);

	sig = sig !2;

	snd1 = Ringz.ar(sig[0], 100, 10);

	env2 = EnvGen.ar(Env.perc(0.01, 5.0, 900), Impulse.ar(0.125));
	snd2 = FreqShift.ar(sig[1], env2 + fbIn.range(10, 50));

	snd = (snd1 * -30.dbamp) + (snd2 * \chimes_amp.ar(5.dbamp).lag(10));
	snd = GVerb.ar(snd, 1, 5, 0.1).sum;
	snd = LPF.ar(snd, 200);


	snd = Splay.ar(snd);

	Out.ar(\fb_out22.ar(40), (snd * -30.dbamp).tanh);
	Out.ar(out_fx, (snd * -30.dbamp).tanh);
	Out.ar(out_chain, (snd * -30.dbamp).tanh);
}).add;
);


//FX-Synth 3
(
SynthDef(\fb1, {
	arg out_fx = 8;
	var sig, fbIn, snd1, env2, snd2, snd3, snd4, snd, randIndex = [0, 0, 0, 0], random = [0, 0, 0, 0];

	sig = [In.ar(26, 1), In.ar(26, 1), In.ar(24, 1) * 10.dbamp, In.ar(24, 1) * 10.dbamp];
	fbIn = In.ar(45);


	~trig0 = TRand.ar(0.0, 4.0, Impulse.ar(0.25));
	~trig1 = TRand.ar(0.0, 4.0, Impulse.ar(0.25));
	~trig2 = TRand.ar(0.0, 4.0, Impulse.ar(0.25));
	~trig3 = TRand.ar(0.0, 4.0, Impulse.ar(0.25));

	sig = Select.ar([~trig0, ~trig1, ~trig2, ~trig3], sig);

	snd1 = Gate.ar(sig[0], SinOsc.ar(\gate_speed.ir(3)));
	snd1 = snd1 * \amp_layer1.ar(1.0).lag(10);

	env2 = EnvGen.ar(Env.perc(), Impulse.ar(fbIn.range(0.25, 1.0)));
	snd2 = Ringz.ar(sig[1] * env2, LFSaw.ar(0.0625).range(200, 500), 1.0).tanh;
	snd2 = BPF.ar(snd2, 900);
	snd2 = snd2 * \amp_layer2.ar(1.0).lag(10);

	snd3 = sig[2] * (Pulse.ar(5000, 0.5, 0.5) * fbIn.range(0.0, 1.0));
	snd3 = snd3 * \amp_layer3.ar(1.0).lag(10);

	snd4 = SmoothDecimator.ar(sig[3], SinOsc.ar(0.125).range(2250, 1125), 0.7);
	snd4 = snd4 * \amp_layer4.ar(1.0).lag(10);



	~trig02 = TRand.ar(0.0, 4.0, Impulse.ar(\change_rate.ar(0.25)));
	~trig12 = TRand.ar(0.0, 4.0, Impulse.ar(\change_rate.ar(0.25)));
	~trig22 = TRand.ar(0.0, 4.0, Impulse.ar(\change_rate.ar(0.25)));
	~trig32 = TRand.ar(0.0, 4.0, Impulse.ar(\change_rate.ar(0.25)));

	//auch der effekte channel (snd1, snd2, snd3, snd4) wird gescrampelt
	snd = Splay.ar(Select.ar([~trig02, ~trig12, ~trig22, ~trig32], [snd1, snd2, snd3, snd4]) * -20.dbamp);


	Out.ar(\fb_out1.ar(45), snd);
	Out.ar(out_fx, snd);
}).add;
);


//FX-Bus 1
(
SynthDef(\subject_fx, {
	arg out = 4;
	var in, reverb, local, del, crush, fx, mix, rm, postFilter;

	in = In.ar(8, 4);

	reverb = GVerb.ar(in, \roomsize.ar(20), \revtime.ar(3), SinOsc.ar(0.2).range(\damp_low.ar(0.9), \damp_high.ar(0.9)));


	local = LocalIn.ar(4) * \del_fb.ar(0.8);
	del = DelayN.ar((in + local), 2.0, \del_time.ar(0.2));
	LocalOut.ar(del);


	fx = (reverb * \rev_level.ar(0.6)) + (del * \del_level.ar(0.0));
	mix = ((fx * \dry_wet.ar(0.0).lag(2)) + (in * \dry_wet.ar(0.0).lag(2).range(1.0, 0.0)));


	mix.tanh;
	Splay.ar(mix);
	// mix = Splay.ar(mix);
	Out.ar(out, mix * \amp.ar(0.9).lag(0.5));
}).add;
);



//FX-Bus 2
(
SynthDef(\contraSubject_fx, {
	arg out = 4;
	var in, reverb, local, del, crush, fx, mix, rm, postFilter;

	in = In.ar(12, 4);

	reverb = GVerb.ar(in, \roomsize.ar(6), \revtime.ar(5), SinOsc.ar(0.2).range(\damp_low.ar(0.9), \damp_high.ar(0.9)));


	local = LocalIn.ar(4) * \del_fb.ar(0.8);
	del = DelayN.ar((in + local), 5.0, \del_time.ar(1.0));
	LocalOut.ar(del);

	crush = Decimator.ar(in, \crush_rate.ar(44100.0), \crush_bits.ar(8));


	fx = (reverb * \rev_level.ar(0.6)) + (del * \del_level.ar(0.0)) + (crush * \crush_level.ar(0.0));
	mix = ((fx * \dry_wet.ar(0.5).lag(2)) + (in * \dry_wet.ar(0.5).lag(2).range(1.0, 0.0)));


	//post_fx
	rm = mix * SinOsc.ar(\ring_freq.ar(200.0));
	mix = SelectX.ar(\post_fx.ar(0), [mix, rm]);

	postFilter = MoogFF.ar(mix, SinOsc.ar(\filter_freq.ar(0.2)).range(1000,5000));
	mix = SelectX.ar(\post_filter.ar(0), [mix, postFilter]);

	mix.tanh;
	Splay.ar(mix);
	// mix = Splay.ar(mix);
	Out.ar(out, mix * \amp.ar(0.9).lag(0.5));
}).add;
);


//"Mixer"
(
SynthDef(\sum, {
	arg out = 0, amp = 0.9;
	var in;

	in = In.ar(4, 4); //der kriegt auf 4 rein - rest muss dann höher

	Out.ar(out, (in * amp).clip(-1, 1));
}).add;
)

)



//Routine/Ablauf des Stückes
(
(
~snd = Synth(\snd);
~fb4 = Synth(\fb4, addAction: 'addToTail');
~fb2 = Synth(\fb2, addAction: 'addToTail');
~fb22 = Synth(\fb22, addAction: 'addToTail');
~fb1 = Synth(\fb1, addAction: 'addToTail');
~subject_fx = Synth(\subject_fx, addAction: 'addToTail');
~contraSubject_fx = Synth(\contraSubject_fx, addAction: 'addToTail');
~sum = Synth(\sum, addAction: 'addToTail');
);


(
Routine {
	~subject_fx.set("amp", 3.0);
	~contraSubject_fx.set("amp", 3.0);


	10.0.wait;
	"DUX1".postln;
	~snd.set("envelope", 1);

	~subject_fx.set("rev_level", 0.6);
	~subject_fx.set("damp_low", 0.1);
	~subject_fx.set("damp_high", 0.5);
	~subject_fx.set("del_level", 0.3);
	~subject_fx.set("dry_wet", 0.5);


	10.0.wait;
	"COMES1".postln;
	~snd.set("out_fx", 12);
	~snd.set("out_chain", 16);


	~snd.set("attack", 0.001);
	~contraSubject_fx.set("amp", 2.4);
	~contraSubject_fx.set("damp_high", 0.7);
	~contraSubject_fx.set("rev_level", 0.3);
	~contraSubject_fx.set("del_time", 0.6);
	~contraSubject_fx.set("del_fb", 0.4);
	~contraSubject_fx.set("crush_level", 0.6);


	~fb4.set();


	10.0.wait;
	~snd.set("out_fx", 50);
	"DUX2".postln;
	~fb4.set("out_fx", 12);
	~fb4.set("out_chain", 18);



	~fb2.set();


	10.0.wait;
	~fb4.set("out_fx", 50);
	"COMES2".postln;

	~fb2.set("out_fx", 12);
	~fb2.set("out_chain", 22);

	~fb22.set();


	10.0.wait;
	~fb2.set("out_fx", 50);
	~fb22.set("out_fx", 50);
	"ZWISCHENSPIEL".postln;
	~fb2.set("out_chain", 26);
	~fb22.set("out_chain", 24);

	~fb1.set();


	// ------------------


	20.0.wait;
	"COMES2_back".postln;
	~fb2.set("out_fx", 12);
	~fb2.set("out_chain", 22);

	~fb22.set("out_fx", 8);


	10.0.wait;
	"DUX2_back".postln;
	~fb4.set("out_fx", 12);

	~fb2.set("out_fx", 8);
	~fb2.set("out_chain", 50);
	~fb22.set("out_chain", 50);


	10.0.wait;
	"COMES1_back".postln;
	~snd.set("out_fx", 12);

	~fb4.set("out_fx", 8);
	~fb4.set("out_chain", 50);


	10.0.wait;
	~fb4.set("out_fx", 50);
	"DUX1_back".postln;
	~snd.set("out_fx", 8);
	~snd.set("out_chain", 50);


}.play;
)
)



