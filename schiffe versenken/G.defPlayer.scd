//=== CONTROLLING OF THE EXTRA SOUNDS ===\\
//pre written comands to get executed on after an other
//(timing is up to performer)

(
Ndef(\lowpad).fadeTime = 30;
Ndef(\lowpad).playN([34, 9, 26, 17]);
)
(
Ndef(\lowpad).fadeTime = 20;
Ndef(\lowpad).set(\hpffreq, 30);
)
(
Ndef(\lowpad).fadeTime = 15;
Ndef(\lowpad).set(\freq, 250);
Ndef(\lowpad).set(\amp, 0.85);
)
(
fork{
	Ndef(\lowpad).fadeTime = 3;
	5.do({
		Ndef(\lowpad).set(\mainfreq, 600);
		Ndef(\lowpad).set(\lpf, 200);
		3.0.wait;
		Ndef(\lowpad).set(\mainfreq, 50);
		15.0.wait;
	})
}
)
(
Ndef(\lowpad).fadeTime = 15;
Ndef(\lowpad).set(\freq, 500);
Ndef(\lowpad).set(\lpf, 1000);
)
(
fork{
	Ndef(\lowpad).fadeTime = 10;
	1.do({
		Ndef(\lowpad).set(\freq, 1000);
		3.0.wait;
		Ndef(\lowpad).set(\freq, 200);
	});
};
Ndef(\sprinkle).fadeTime = 25;
Ndef(\sprinkle).playN([49, 51, 55, 57]);//wider
)
(
Ndef(\sprinkle).fadeTime = 15;
Ndef(\sprinkle).set(\envatt, 0.1);
)
(
Ndef(\sprinkle).fadeTime = 20;
Ndef(\sprinkle).set(\pratio, 4);
Ndef(\sprinkle).set(\lpf, 5000);
Ndef(\sprinkle).set(\amp, 0.5);
)
(
Ndef(\sprinkle).set(\trig, 0.6);
Ndef(\sprinkle).set(\amp, 0.2);
)
(
Ndef(\sprinkle).set(\envrel, 0.8);
)
(
Ndef(\sprinkle).fadeTime = 15;
fork{
	3.do({
		Ndef(\sprinkle).set(\freq, 500);
		5.0.wait;
		Ndef(\sprinkle).set(\freq, 200);
		5.0.wait;
		Ndef(\sprinkle).set(\freq, 1000);
	})
}
Ndef(\sprinkle).set(\freq, 500);
Ndef(\sprinkle).set(\amp, 0.1);
)
(
Ndef(\sprinkle).set(\envatt, 0.3);
Ndef(\sprinkle).set(\envrel, 0.5);
Ndef(\sprinkle).set(\trig, 0.3);
)
(
Ndef(\bassy).fadeTime = 30;
Ndef(\bassy).play([35, 38, 45, 42]);
)
(
Ndef(\lowpad).stop(30);
Ndef(\bassy).set(\hpffreq, 50);
)
(
Ndef(\bassy).fadeTime = 15;
Ndef(\bassy).set(\shift, 3);
Ndef(\bassy).set(\duration, 0.7);
)
(
Ndef(\bassy).set(\filtermod, 1);
Ndef(\pad).fadeTime = 40;
Ndef(\pad).play({63.rand}!15);
Ndef(\sprinkle).stop(60);
)
(
Ndef(\bassy).fadeTime = 25;
Ndef(\bassy).set(\trigfreq, 0.8);
Ndef(\bassy).set(\bits, 3);
Ndef(\bassy).set(\amp, 0.8);
Ndef(\pad).fadeTime = 25;
Ndef(\pad).set(\density, 5);
)
(
Ndef(\bassy).fadeTime = 15;
Ndef(\bassy).set(\lpffreq, 150);
Ndef(\pad).set(\lpffreq, 500);
Ndef(\pad).set(\wetverb, 1);
Ndef(\pad).set(\amp, 0.4);
)
(
Ndef(\pad).set(\verbmod, 30);
Ndef(\pad).set(\amp, 0.2);
Ndef(\pad).set(\dryverb, 0.5);
Ndef(\pad).set(\density, 30);
)
(
Ndef(\pad).set(\dryverb, 0);
Ndef(\pad).set(\pulsemod, 1);
Ndef(\pad).set(\pulsefreq, 80);
)
(
Ndef(\pad).set(\dryverb, 0.1);
Ndef(\pad).set(\bpf, 1);
Ndef(\pad).set(\density, 500);
Ndef(\pad).set(\amp, 0.1);
Ndef(\pad).set(\verbmod, 300);
)
(
Ndef(\texture).fadeTime = 30;
Ndef(\texture).play([34, 9, 26, 17, 30, 13]);
)
(
Ndef(\texture).fadeTime = 30;
Ndef(\texture).set(\hpf, 150);
Ndef(\bassy).stop(35);
fork{
	3.do({
		Ndef(\pad).set(\density, 150);
		5.0.wait;
		Ndef(\pad).set(\verbmod, 500);
		3.0.wait;
		Ndef(\pad).set(\density, 25);
		Ndef(\pad).set(\verbmod, 0.3);
		9.0.wait;
	})
}
Ndef(\pad).set(\density, 450);
Ndef(\pad).set(\verbmod, 350);
)
(
Ndef(\texture).fadeTime = 15;
Ndef(\texture).set(\envatt, 0.1);
Ndef(\texture).set(\envrel, 2.5);
Ndef(\pad).set(\density, 350);
)
(
Ndef(\texture).set(\pulsemod, 0.7);
Ndef(\texture).set(\decaytime, 5.7);
Ndef(\texture).set(\amp, 0.2);
Ndef(\texture).set(\hpf, 150);
Ndef(\pad).set(\density, 250);
)
(
Ndef(\texture).set(\envatt, 0.4);
Ndef(\texture).set(\im, 3.4);
Ndef(\texture).set(\amp, 0.06);
Ndef(\pad).set(\density, 150);
)
(
Ndef(\texture).set(\pulsemod, 5);
Ndef(\texture).set(\decaytime, 1.7);
Ndef(\texture).set(\amp, 0.1);
Ndef(\pad).set(\density, 50);
)
(
Ndef(\texture).set(\im, 0.4);
Ndef(\texture).fadeTime = 60;
Ndef(\texture).set(\filterfreqlo, 1000);
Ndef(\pad).fadeTime = 90;
Ndef(\pad).set(\density, 4);
)
(
Ndef(\pad).stop(15)
)
(
Ndef(\texture).stop(2)
)

