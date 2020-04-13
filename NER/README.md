# BumbleBee
<img src="https://user-images.githubusercontent.com/25641555/76114333-d7a63480-5fb3-11ea-96e1-8d2ff27c4a7f.png" width="128" height="200" />

Natural Language Processing , LSTM , CNN, BERT, NER

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		The Best Results are represented in the SOAT (State Of The Art). The Recall, Precision and F1-score related to each architecture with different Embedding layers represented.
	</caption>
	<tr>
		<td style="border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td colspan="4" style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Pararrel BiLSTM-BiCRF
		</td>
		<td colspan="4" style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BiLSTM-BiCRF
		</td>
		<td colspan="4" style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Sequence BiLSTM-BiCRF
		</td>
		<td colspan="4" style="border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BiLSTM-CRF
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			SOTA
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BIO
			<br>
			BERT
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			RNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			CNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Look
			<br>
			up
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			SAE
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			RNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			CNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Look
			<br>
			up
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			SAE
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			RNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			CNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Look
			<br>
			up
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			SAE
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			RNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			CNN
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Look
			<br>
			up
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			SAE
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BC2GM
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.61
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.60
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.90 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.37
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.62
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.62
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.87 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.61
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.60
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.87 </b>
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.39
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BC5CDR
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.67
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.69
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.48
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.52
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.39
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.89 </b>
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.60
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.56
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.33
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.75
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.33
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.33
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.79 </b>
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.33
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.73
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.68
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.68
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.59
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.57
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.31
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.31
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.31
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.81 </b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.31
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.73
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.74
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.71
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			BioNLP13PC
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.88 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.71
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.82 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.68
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.59
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.65
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.70
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.84 </b>
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.79
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.69
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.63
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.68
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.72
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Linnaeus
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.98
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.96
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.65
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.95
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.96 </b>
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.95 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.97
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.98
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.94
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.56
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.48
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.71
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.53
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.44
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.50
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.73 </b>
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.51
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.77 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.51
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.59
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.60
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.93
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.59
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.52
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.75
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.56
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.47
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.56
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.75
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.41
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.76 </b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.56
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.84 </b>
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.59
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.67
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.68
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			NCBI
			<br>
			DISEASE
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.93
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.64
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.91
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.93
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.38
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.91
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.38
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.93 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.49
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.91
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.92
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.62
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.89 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.86
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.42
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.90
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.62
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.89
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.40
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.40
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.90 </b>
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.88
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.40
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.85
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.87
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.86~
		</td>
	</tr>
	<tr>
		<td rowspan="3" style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			JNLPBA
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			P
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.74
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.69
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.83 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.74
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.72
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.74
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
	</tr>
	<tr>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			R
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.84
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.77
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			<b> 0.84 </b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-right: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.76
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.75
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.73
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt;">
			0.72
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.78
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.82
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.83
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.80
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.72
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			<b> 0.83 </b>
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.81
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.74
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.72
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.73
		</td>
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			0.73~
		</td>
	</tr>
</table>
Natural Language Processing , LSTM , CNN, NER
